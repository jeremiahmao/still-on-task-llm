"""Phase 3: Ablations + deferred methods (if compute permits).

Run after Phases 1-2 produce core results.
"""

import subprocess
import sys

ABLATIONS = [
    # Mixed replay (deferred from Phase 1)
    {
        "method": "mixed_replay",
        "scale": 1000,
        "config": "configs/update/mixed_replay.yaml",
        "overrides": [],
        "tag": "mixed_replay_1K",
    },
    # COPR ablations: replay percentage
    {
        "method": "copr",
        "scale": 1000,
        "config": "configs/update/copr.yaml",
        "overrides": ["replay_pct=0.01"],
        "tag": "copr_replay1pct",
    },
    {
        "method": "copr",
        "scale": 1000,
        "config": "configs/update/copr.yaml",
        "overrides": ["replay_pct=0.10"],
        "tag": "copr_replay10pct",
    },
    # COPR ablations: K (number of sampled responses)
    {
        "method": "copr",
        "scale": 1000,
        "config": "configs/update/copr.yaml",
        "overrides": ["K=4"],
        "tag": "copr_K4",
    },
    # KL-reg lambda sweep
    {
        "method": "kl_reg_sft",
        "scale": 1000,
        "config": "configs/update/kl_reg_sft.yaml",
        "overrides": ["kl_lambda=0.01"],
        "tag": "kl_lambda001",
    },
    {
        "method": "kl_reg_sft",
        "scale": 1000,
        "config": "configs/update/kl_reg_sft.yaml",
        "overrides": ["kl_lambda=1.0"],
        "tag": "kl_lambda1",
    },
    # Full retrain (most expensive — run last)
    {
        "method": "full_retrain",
        "scale": 1000,
        "config": None,
        "overrides": [],
        "tag": "full_retrain_1K",
    },
]


def main():
    for i, ablation in enumerate(ABLATIONS):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(ABLATIONS)}] {ablation['tag']}")
        print(f"{'=' * 60}")

        cmd = [
            sys.executable,
            "scripts/09_run_update.py",
            "--method",
            ablation["method"],
            "--scale",
            str(ablation["scale"]),
            "--task",
            "qd",
        ]
        if ablation["config"]:
            cmd.extend(["--config", ablation["config"]])
        if ablation["overrides"]:
            cmd.extend(["--overrides"] + ablation["overrides"])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {ablation['tag']}")
            continue

        run_id = f"{ablation['method']}_qd_scale{ablation['scale']}"
        eval_cmd = [
            sys.executable,
            "scripts/10_evaluate.py",
            "--model_path",
            f"outputs/{run_id}",
            "--task",
            "qd",
        ]
        subprocess.run(eval_cmd)

    print(f"\nPhase 3 complete. {len(ABLATIONS)} ablations run.")


if __name__ == "__main__":
    main()
