"""Stratified Spectral LoRA (SSL) — teacher-free architectural preservation.

The novel ingredient is preservation-without-a-teacher: instead of computing KL
against a frozen reference snapshot (DSAE Lite), preservation comes from
(a) initializing each LoRA at a layer-specific position in its base weight's
singular spectrum and (b) scaling the per-layer learning rate inversely to that
layer's input/output similarity. Pass-through layers (high I/O cosine similarity)
get their LoRA placed at bottom singular components AND a low LR — they
barely move. Transformative layers (low similarity, e.g. mid-MLP) get
intermediate-spectrum init AND full LR — they carry the new fact gradient.

Combines two 2025 ingredients:
- Dalvi et al. (2602.03493): the U-shaped forgetting curve across the singular
  value spectrum — intermediate components give the best forgetting/performance
  trade-off vs. PiSSA-style top init or MiLoRA-style bottom init.
- Red Hat AI / Sculpting Subspaces (2504.07097): activation-similarity-based
  per-layer importance.

The cross-pollination — picking *spectral position* per layer based on
*activation similarity* — has not been published. See
paper/ml_intern_arch_method.md §5 for novelty discussion.

Operates on K=5 augmented injection data (same as DSAE Lite / aug_sft_k5).
No teacher model. No KL. No replay buffer. The architecture itself is the
preservation mechanism.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from sot.update.base import UpdateMethod


def _is_peft_lora_linear(module) -> bool:
    """Return True for PEFT LoraLayer wrapping a Linear (has base_layer +
    lora_A/lora_B ModuleDicts)."""
    return (
        hasattr(module, "base_layer")
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
    )


def _adapter_key(module) -> str:
    """PEFT keys lora_A / lora_B by adapter name. We use the active one."""
    if hasattr(module, "active_adapters") and module.active_adapters:
        return module.active_adapters[0]
    return "default"


def _extract_user_question(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") == "user":
            return m.get("content")
    return None


class SSLUpdate(UpdateMethod):
    @property
    def name(self) -> str:
        return "ssl_inject"

    # --------------------------------------------------------------------------
    # Public entry point
    # --------------------------------------------------------------------------

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        cfg = cfg or DictConfig({})
        n_calib = cfg.get("calibration_samples", 256)
        ssl_alpha = cfg.get("ssl_alpha", 1.0)
        # Spectral mode controls where each layer's LoRA is initialized:
        # - "adaptive": s = floor(I_l * (d - r))   (THE RECOMMENDED MODE)
        # - "top":      s = 0                       (PiSSA-style; FM2 ablation)
        # - "mid":      s = (d - r) // 2            (uniform mid; FM2 ablation)
        # - "bottom":   s = d - r                   (MiLoRA-style; FM2 ablation)
        # - "uniform":  skip SVD, leave PEFT random init  (per-layer-LR-only fallback)
        spectral_mode = cfg.get("spectral_mode", "adaptive")
        base_lr = cfg.get("training", {}).get("lr", 2e-5)
        epochs = cfg.get("training", {}).get("epochs", 3)
        batch_size = cfg.get("training", {}).get("batch_size", 8)
        max_seq_length = cfg.get("training", {}).get("max_seq_length", 512)
        log_per_layer = cfg.get("log_per_layer", True)

        lora_layers = {
            name: module
            for name, module in model.named_modules()
            if _is_peft_lora_linear(module)
        }
        if not lora_layers:
            raise RuntimeError(
                "SSLUpdate: no PEFT LoraLayer found. SSL requires PEFT LoRA "
                "to be applied before method.apply() is called."
            )
        print(f"  SSL: found {len(lora_layers)} LoRA-wrapped layers")

        # 1. Calibrate per-layer activation similarity I^(l).
        layer_sim = self._calibrate(
            model, tokenizer, lora_layers, task_data, fact_qa_pairs,
            n_calib=n_calib, max_seq_length=max_seq_length,
        )

        # 2. Override LoRA init at the per-layer spectral position
        #    (and subtract from base weights to keep initial equivalence).
        per_layer_pos = self._init_spectral(
            model, lora_layers, layer_sim, spectral_mode=spectral_mode,
        )

        # 3. Build optimizer with per-layer LR.
        optimizer, per_layer_lr = self._build_optimizer(
            lora_layers, layer_sim, base_lr=base_lr, ssl_alpha=ssl_alpha,
        )

        if log_per_layer:
            log_path = Path(cfg.get("output_dir", "./checkpoints/ssl_inject"))
            log_path.mkdir(parents=True, exist_ok=True)
            with open(log_path / "ssl_calibration.json", "w") as f:
                json.dump(
                    {
                        "spectral_mode": spectral_mode,
                        "ssl_alpha": ssl_alpha,
                        "base_lr": base_lr,
                        "layer_sim": layer_sim,
                        "spectral_position": per_layer_pos,
                        "per_layer_lr": per_layer_lr,
                    },
                    f, indent=2,
                )
            print(f"  Wrote SSL calibration log -> {log_path / 'ssl_calibration.json'}")

        # 4. Standard SFT on K=5 augmented data (same dispatch as kl_reg_sft).
        fact_texts = self._render_fact_texts(tokenizer, fact_qa_pairs)

        def collate_fn(batch_texts):
            orig_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "right"
            result = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_seq_length,
            )
            tokenizer.padding_side = orig_padding_side
            return result

        fact_loader = DataLoader(
            fact_texts, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        )

        model.train()
        for epoch in range(epochs):
            total = 0.0
            n = 0
            optimizer.zero_grad()
            for batch in tqdm(fact_loader, desc=f"SSL epoch {epoch + 1}/{epochs}"):
                first_device = next(model.parameters()).device
                batch = {k: v.to(first_device) for k, v in batch.items()}
                labels = batch["input_ids"].clone()
                labels[batch["attention_mask"] == 0] = -100

                outputs = model(**batch, labels=labels)
                loss = outputs.loss
                total += loss.item()
                n += 1
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f"  Epoch {epoch + 1}: sft_loss={total / max(n, 1):.4f}")

        return model

    # --------------------------------------------------------------------------
    # Calibration: compute I^(l) for each LoRA-wrapped layer
    # --------------------------------------------------------------------------

    def _calibrate(
        self, model, tokenizer, lora_layers, task_data, fact_qa_pairs,
        n_calib, max_seq_length,
    ) -> dict[str, float]:
        """Compute per-layer mean cosine similarity between layer input and
        layer output across n_calib calibration samples."""
        # Prefer task replay data; fall back to fact-render data if absent.
        if task_data:
            calib_texts = []
            for item in task_data[:n_calib * 4]:
                q = _extract_user_question(item.get("messages", []))
                if q:
                    calib_texts.append(
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": q}],
                            tokenize=False, add_generation_prompt=True,
                        )
                    )
                if len(calib_texts) >= n_calib:
                    break
        else:
            calib_texts = []
        if len(calib_texts) < n_calib:
            # Top up with rendered facts (raw chat format).
            for qa in fact_qa_pairs[: (n_calib - len(calib_texts))]:
                if qa.get("chat_messages"):
                    chat = qa["chat_messages"]
                else:
                    chat = [
                        {"role": "user", "content": qa.get("question", "")},
                    ]
                calib_texts.append(
                    tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True,
                    )
                )
        calib_texts = calib_texts[:n_calib]
        print(f"  SSL calibration: {len(calib_texts)} samples")

        # Hook each LoRA-wrapped layer's forward to record (input, output).
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        hooks = []

        def make_hook(layer_name):
            def hook(module, inputs, output):
                # inputs is a tuple; the first element is the hidden state.
                x = inputs[0]
                y = output if not isinstance(output, tuple) else output[0]
                # Flatten batch+seq, mean cosine sim across token positions.
                x2 = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
                y2 = y.detach().to(torch.float32).reshape(-1, y.shape[-1])
                # If shapes differ (e.g. up_proj: in_dim != out_dim), skip —
                # cosine similarity needs matching dims. Per design, only
                # square-ish projections (q/k/v/o, gate-vs-up not, down) have
                # matching in/out dims. For mismatched, we use a different
                # proxy: norm ratio (||y|| / ||x||) clipped to [0, 1].
                if x2.shape[-1] != y2.shape[-1]:
                    rx = x2.norm(dim=-1)
                    ry = y2.norm(dim=-1)
                    score = (ry / (rx + 1e-9)).clamp(0.0, 1.0).mean().item()
                else:
                    score = F.cosine_similarity(x2, y2, dim=-1).mean().item()
                sums[layer_name] += score
                counts[layer_name] += 1
            return hook

        for name, module in lora_layers.items():
            hooks.append(module.register_forward_hook(make_hook(name)))

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for txt in calib_texts:
                    enc = tokenizer(
                        txt, return_tensors="pt", truncation=True,
                        max_length=max_seq_length,
                    ).to(device)
                    model(**enc)
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                model.train()

        layer_sim = {n: sums[n] / max(counts[n], 1) for n in lora_layers}
        # Normalize so the mean is 1.0 (per spec §3 in arch_method.md).
        if layer_sim:
            mean_sim = sum(layer_sim.values()) / len(layer_sim)
            if mean_sim > 1e-6:
                layer_sim = {n: v / mean_sim for n, v in layer_sim.items()}
        return layer_sim

    # --------------------------------------------------------------------------
    # Spectral init: PiSSA-style at per-layer position s^(l)
    # --------------------------------------------------------------------------

    def _init_spectral(
        self, model, lora_layers, layer_sim, spectral_mode,
    ) -> dict[str, int]:
        """Re-initialize each LoRA's A,B from SVD components at the spectral
        position implied by I^(l) (or by the fixed mode for ablations).
        Subtracts the resulting initial ΔW from the base layer weight so that
        the model's effective forward at step 0 equals the original base
        forward (same trick as PiSSA)."""
        per_layer_pos: dict[str, int] = {}
        if spectral_mode == "uniform":
            # Skip SVD init entirely — leave PEFT's random init in place.
            # This is the "per-layer-LR-only" ablation fallback.
            return per_layer_pos

        for name, module in tqdm(lora_layers.items(), desc="  SSL spectral init"):
            base = module.base_layer
            W = base.weight.data
            adapter = _adapter_key(module)
            lora_A_param = module.lora_A[adapter].weight  # shape (r, in)
            lora_B_param = module.lora_B[adapter].weight  # shape (out, r)
            r = lora_A_param.shape[0]
            assert lora_B_param.shape[1] == r

            d = min(W.shape)
            r_clamped = min(r, d)

            # Pick spectral start position.
            if spectral_mode == "adaptive":
                I_l = layer_sim.get(name, 1.0)
                s = int(I_l * (d - r_clamped))
            elif spectral_mode == "top":
                s = 0
            elif spectral_mode == "mid":
                s = (d - r_clamped) // 2
            elif spectral_mode == "bottom":
                s = max(0, d - r_clamped)
            else:
                raise ValueError(f"unknown spectral_mode: {spectral_mode}")
            s = max(0, min(s, d - r_clamped))
            per_layer_pos[name] = s

            # SVD in fp32 for numerical stability.
            W_fp32 = W.detach().to(torch.float32)
            U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)
            sqrt_S = S[s : s + r_clamped].clamp(min=0.0).sqrt()
            # PEFT convention: forward computes
            #     scale * lora_B @ lora_A @ x   where scale = alpha / r.
            # We want scale * lora_B @ lora_A == U[:, s:s+r] @ diag(S[s:s+r]) @ Vh[s:s+r, :]
            # so set lora_A = (1/sqrt(scale)) * sqrt(S) * Vh and lora_B = (1/sqrt(scale)) * U * sqrt(S).
            scale = float(module.scaling.get(adapter, 1.0)) if hasattr(module, "scaling") else 1.0
            scale = max(scale, 1e-9)
            inv_sqrt_scale = 1.0 / math.sqrt(scale)
            new_A = (sqrt_S.unsqueeze(1) * Vh[s : s + r_clamped, :]) * inv_sqrt_scale
            new_B = (U[:, s : s + r_clamped] * sqrt_S.unsqueeze(0)) * inv_sqrt_scale

            # Pad if r > d (shouldn't happen with r=16 < d~3584 but defensive).
            if r_clamped < r:
                pad_A = torch.zeros(r - r_clamped, new_A.shape[1], dtype=new_A.dtype)
                pad_B = torch.zeros(new_B.shape[0], r - r_clamped, dtype=new_B.dtype)
                new_A = torch.cat([new_A, pad_A], dim=0)
                new_B = torch.cat([new_B, pad_B], dim=1)

            # Effective ΔW that PEFT will add to base at forward time.
            delta_W = scale * (new_B @ new_A)

            tgt_dtype = W.dtype
            tgt_device = W.device
            with torch.no_grad():
                lora_A_param.data.copy_(new_A.to(tgt_dtype).to(tgt_device))
                lora_B_param.data.copy_(new_B.to(tgt_dtype).to(tgt_device))
                # Subtract initial delta from base so initial effective W == W.
                base.weight.data.sub_(delta_W.to(tgt_dtype).to(tgt_device))

        return per_layer_pos

    # --------------------------------------------------------------------------
    # Optimizer: per-layer learning rate
    # --------------------------------------------------------------------------

    def _build_optimizer(
        self, lora_layers, layer_sim, base_lr, ssl_alpha,
    ):
        param_groups = []
        per_layer_lr: dict[str, float] = {}
        for name, module in lora_layers.items():
            adapter = _adapter_key(module)
            params = [
                module.lora_A[adapter].weight,
                module.lora_B[adapter].weight,
            ]
            I_l = layer_sim.get(name, 1.0)
            # η^(l) = η_base * min(1, α / I^(l))
            #
            # Layers with I_l > α have reduced LR: pass-through layers
            # (high I/O similarity) train slowly. Layers with I_l < α get
            # full base LR.
            if I_l < 1e-6:
                lr_l = base_lr
            else:
                lr_l = base_lr * min(1.0, ssl_alpha / I_l)
            param_groups.append({"params": params, "lr": lr_l})
            per_layer_lr[name] = lr_l
        return torch.optim.AdamW(param_groups), per_layer_lr

    # --------------------------------------------------------------------------
    # Fact rendering (mirrors KLRegSFTUpdate / NaiveSFTUpdate dispatch)
    # --------------------------------------------------------------------------

    def _render_fact_texts(self, tokenizer, fact_qa_pairs) -> list[str]:
        out: list[str] = []
        for qa in fact_qa_pairs:
            if qa.get("chat_messages"):
                chat = qa["chat_messages"]
            elif qa.get("train_format") == "qd" and qa.get("qd_messages"):
                chat = qa["qd_messages"]
            else:
                chat = [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]},
                ]
            out.append(
                tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=False,
                )
            )
        return out
