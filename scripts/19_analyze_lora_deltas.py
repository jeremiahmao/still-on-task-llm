"""Phase 6: analyze LoRA weight deltas across update methods (CPU-only).

For each adapter directory we:
  1. Load adapter A/B matrices from its state dict.
  2. For every (layer, target_module), reconstruct Delta W = (alpha / r) * B @ A.
  3. Report per-layer Frobenius norm and effective rank (stable-rank proxy).
  4. Cross-method: pairwise principal angles between the *input-side* row
     subspaces of Delta W (top-k right singular vectors). Input subspaces
     describe which directions in activation space the update attends to;
     column-space (output) comparisons are a different question.

Usage:
  uv run python scripts/19_analyze_lora_deltas.py \
    --adapters outputs/naive_sft_qd_scale1000/update_adapter \
               outputs/kl_reg_sft_qd_scale1000/update_adapter \
               outputs/copr_qd_scale1000/update_adapter \
    --out outputs/mechanistic/lora_analysis.json

Outputs:
  outputs/mechanistic/lora_analysis.json   per-method, per-layer statistics + pairwise angles
  outputs/mechanistic/norms_by_layer.png   optional matplotlib figure (if matplotlib is available)
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_adapter_state(adapter_dir: Path) -> dict[str, np.ndarray]:
    """Load LoRA A/B tensors from safetensors or pytorch_model.bin as numpy arrays."""
    st_path = adapter_dir / "adapter_model.safetensors"
    if st_path.exists():
        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to read adapter_model.safetensors."
            ) from exc
        return load_file(str(st_path))

    pt_path = adapter_dir / "adapter_model.bin"
    if pt_path.exists():
        import torch

        state = torch.load(pt_path, map_location="cpu")
        return {k: v.detach().to(torch.float32).numpy() for k, v in state.items()}

    raise FileNotFoundError(
        f"No adapter weights found in {adapter_dir} (looked for .safetensors and .bin)."
    )


def _load_scaling(adapter_dir: Path) -> float:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return 1.0
    cfg = json.loads(cfg_path.read_text())
    alpha = cfg.get("lora_alpha", 1)
    r = cfg.get("r", 1)
    return float(alpha) / max(int(r), 1)


# Match "...layers.<k>.<mha-or-mlp>.<module>.lora_(A|B)..."
_KEY_RE = re.compile(
    r"layers\.(\d+)\.(?:self_attn|mlp)\.(?P<module>\w+)\."
    r"lora_(?P<which>[AB])\..*"
)


def _extract_lora_pairs(state: dict[str, np.ndarray]):
    """Return dict keyed by (layer, module) -> {"A": ndarray, "B": ndarray}."""
    pairs: dict[tuple[int, str], dict[str, np.ndarray]] = defaultdict(dict)
    for key, tensor in state.items():
        m = _KEY_RE.search(key)
        if not m:
            continue
        layer = int(m.group(1))
        module = m.group("module")
        which = m.group("which")
        pairs[(layer, module)][which] = tensor.astype(np.float32, copy=False)
    return pairs


def _delta_weight(pair: dict[str, np.ndarray], scaling: float) -> np.ndarray | None:
    a = pair.get("A")
    b = pair.get("B")
    if a is None or b is None:
        return None
    # PEFT convention: A is (r, in), B is (out, r) -> BA is (out, in).
    delta = scaling * (b @ a)
    return delta


def _effective_rank(sv: np.ndarray, eps: float = 1e-8) -> float:
    """Shannon-entropy effective rank from singular values (softmax of s^2)."""
    s2 = sv**2
    total = s2.sum()
    if total < eps:
        return 0.0
    p = s2 / total
    p = np.clip(p, eps, 1.0)
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def _stable_rank(sv: np.ndarray, eps: float = 1e-12) -> float:
    """||M||_F^2 / ||M||_2^2 — noise-robust dimensionality proxy."""
    s2 = sv**2
    denom = sv.max() ** 2 if sv.size else eps
    if denom < eps:
        return 0.0
    return float(s2.sum() / denom)


def _principal_angles(U1: np.ndarray, U2: np.ndarray, k: int) -> np.ndarray:
    """First k principal angles (radians) between column spans of U1, U2."""
    k = min(k, U1.shape[1], U2.shape[1])
    if k == 0:
        return np.array([])
    Q1, _ = np.linalg.qr(U1[:, :k])
    Q2, _ = np.linalg.qr(U2[:, :k])
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def analyze_adapter(adapter_dir: Path, subspace_k: int = 8):
    state = _load_adapter_state(adapter_dir)
    scaling = _load_scaling(adapter_dir)
    pairs = _extract_lora_pairs(state)

    per_layer: list[dict] = []
    per_module_norms: dict[str, list[float]] = defaultdict(list)
    subspaces: dict[tuple[int, str], np.ndarray] = {}

    for (layer, module), pair in sorted(pairs.items()):
        delta = _delta_weight(pair, scaling)
        if delta is None:
            continue
        fro = float(np.linalg.norm(delta, ord="fro"))
        try:
            sv = np.linalg.svd(delta, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        per_layer.append(
            {
                "layer": layer,
                "module": module,
                "fro_norm": fro,
                "spectral_norm": float(sv.max()) if sv.size else 0.0,
                "stable_rank": _stable_rank(sv),
                "effective_rank": _effective_rank(sv),
            }
        )
        per_module_norms[module].append(fro)
        # Save the top-k right singular vectors (rows of Vh) so we compare
        # the input/row subspace of Delta W — i.e. the directions in activation
        # space that the update acts on.
        if subspace_k > 0:
            try:
                _, _, Vh = np.linalg.svd(delta, full_matrices=False)
                subspaces[(layer, module)] = Vh[:subspace_k].T.copy()
            except np.linalg.LinAlgError:
                pass

    summary = {
        "adapter_dir": str(adapter_dir),
        "lora_scaling": scaling,
        "n_pairs": len(per_layer),
        "per_layer": per_layer,
        "total_fro_norm": float(np.sqrt(sum(p["fro_norm"] ** 2 for p in per_layer))),
        "mean_stable_rank": float(np.mean([p["stable_rank"] for p in per_layer]))
        if per_layer
        else 0.0,
        "mean_effective_rank": float(np.mean([p["effective_rank"] for p in per_layer]))
        if per_layer
        else 0.0,
        "per_module_mean_fro": {m: float(np.mean(v)) for m, v in per_module_norms.items()},
    }
    return summary, subspaces


def pairwise_subspace_overlap(
    subspaces_by_method: dict[str, dict[tuple[int, str], np.ndarray]],
    k: int = 8,
):
    """Mean cos(angle) between input/row subspaces across shared (layer, module) sites."""
    methods = list(subspaces_by_method.keys())
    result = {}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            s1 = subspaces_by_method[m1]
            s2 = subspaces_by_method[m2]
            shared = sorted(set(s1.keys()) & set(s2.keys()))
            if not shared:
                continue
            mean_cos_first = []
            mean_cos_all = []
            for key in shared:
                angles = _principal_angles(s1[key], s2[key], k)
                if angles.size == 0:
                    continue
                mean_cos_first.append(float(np.cos(angles[0])))
                mean_cos_all.append(float(np.cos(angles).mean()))
            if not mean_cos_first:
                continue
            result[f"{m1}__vs__{m2}"] = {
                "n_shared_sites": len(shared),
                "mean_cos_first_angle": float(np.mean(mean_cos_first)),
                "mean_cos_all_angles": float(np.mean(mean_cos_all)),
            }
    return result


def _maybe_plot(summaries: dict[str, dict], out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure.")
        return False

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, summary in summaries.items():
        by_layer = defaultdict(float)
        for entry in summary["per_layer"]:
            by_layer[entry["layer"]] += entry["fro_norm"] ** 2
        layers = sorted(by_layer.keys())
        norms = [np.sqrt(by_layer[l]) for l in layers]
        ax.plot(layers, norms, marker="o", label=method)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("||Delta W|| (Frobenius)")
    ax.set_title("LoRA update magnitude per layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Figure -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapters", nargs="+", required=True,
        help="One or more paths like outputs/<run>/update_adapter",
    )
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="Optional display names per adapter (same order). Defaults to parent run id.",
    )
    parser.add_argument(
        "--out", default="outputs/mechanistic/lora_analysis.json",
        help="Output JSON path.",
    )
    parser.add_argument("--subspace-k", type=int, default=8)
    args = parser.parse_args()

    adapter_paths = [Path(p) for p in args.adapters]
    names = args.names or [p.parent.name for p in adapter_paths]
    if len(names) != len(adapter_paths):
        raise SystemExit("--names must match --adapters length.")

    summaries: dict[str, dict] = {}
    subspaces_by_method: dict[str, dict[tuple[int, str], np.ndarray]] = {}

    for name, path in zip(names, adapter_paths):
        print(f"Analyzing {name}  <-  {path}")
        summary, subspaces = analyze_adapter(path, subspace_k=args.subspace_k)
        summaries[name] = summary
        subspaces_by_method[name] = subspaces

    pairs = pairwise_subspace_overlap(subspaces_by_method, k=args.subspace_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"per_method": summaries, "pairwise_subspace_overlap": pairs}, f, indent=2)
    print(f"JSON -> {out_path}")

    _maybe_plot(summaries, out_path.with_suffix("").with_name("norms_by_layer.png"))


if __name__ == "__main__":
    main()
