"""Phase 3: Full retrain + ablations (if compute permits)."""

import subprocess
import sys


ABLATIONS = [
    # COPR ablations: replay percentage
    {"method": "copr", "scale": 1000, "overrides": ["replay_pct=0.01"], "tag": "copr_replay1pct"},
    {"method": "copr", "scale": 1000, "overrides": ["replay_pct=0.10"], "tag": "copr_replay10pct"},
    # COPR ablations: K (number of sampled responses)
    {"method": "copr", "scale": 1000, "overrides": ["K=4"], "tag": "copr_K4"},
    {"method": "copr", "scale": 1000, "overrides": ["K=16"], "tag": "copr_K16"},
    # KL-reg lambda sweep
    {"method": "kl_reg_sft", "scale": 1000, "overrides": ["kl_lambda=0.01"], "tag": "kl_lambda001"},
    {"method": "kl_reg_sft", "scale": 1000, "overrides": ["kl_lambda=1.0"], "tag": "kl_lambda1"},
    # Full retrain at all scales
    {"method": "full_retrain", "scale": 200, "overrides": [], "tag": "full_retrain_200"},
    {"method": "full_retrain", "scale": 1000, "overrides": [], "tag": "full_retrain_1000"},
    {"method": "full_retrain", "scale": 3000, "overrides": [], "tag": "full_retrain_3000"},
]

CONFIG_MAP = {
    "copr": "configs/update/copr.yaml",
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "full_retrain": None,
}


def main():
    for i, ablation in enumerate(ABLATIONS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(ABLATIONS)}] {ablation['tag']}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "scripts/09_run_update.py",
            "--method", ablation["method"],
            "--scale", str(ablation["scale"]),
            "--task", "qd",
        ]
        config = CONFIG_MAP.get(ablation["method"])
        if config:
            cmd.extend(["--config", config])
        if ablation["overrides"]:
            cmd.extend(["--overrides"] + ablation["overrides"])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {ablation['tag']}")
            continue

        run_id = f"{ablation['method']}_qd_scale{ablation['scale']}"
        eval_cmd = [
            sys.executable, "scripts/10_evaluate.py",
            "--model_path", f"outputs/{run_id}",
            "--task", "qd",
        ]
        subprocess.run(eval_cmd)

    print(f"\nPhase 3 complete. {len(ABLATIONS)} ablations run.")


if __name__ == "__main__":
    main()
