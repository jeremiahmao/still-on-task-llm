"""SageMaker training entry point.

Usage on SageMaker:
    The estimator calls this script with --phase and optional --methods/--scale args.
    Data is expected at /opt/ml/input/data/{fnspid,finqa,qd}/ (SageMaker channels)
    or downloaded at runtime.

    Checkpoints go to /opt/ml/checkpoints/ (persisted across spot restarts).
    Final model goes to /opt/ml/model/ (uploaded to S3 on completion).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# SageMaker paths
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
SM_CHANNEL_DATA = os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data")
SM_CHECKPOINT_DIR = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")

# Repo root (git clone lands here)
REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_paths():
    """Symlink SageMaker paths into the repo's expected locations."""
    repo = REPO_ROOT

    # If SageMaker provided data via channels, symlink into repo
    sm_data = Path(SM_CHANNEL_DATA)
    repo_data = repo / "data"
    if sm_data.exists() and not repo_data.exists():
        repo_data.symlink_to(sm_data)
        print(f"Linked data: {sm_data} -> {repo_data}")

    # Checkpoints: use SageMaker checkpoint dir for spot instance resilience
    sm_ckpt = Path(SM_CHECKPOINT_DIR)
    repo_ckpt = repo / "checkpoints"
    if sm_ckpt.exists():
        sm_ckpt.mkdir(parents=True, exist_ok=True)
        if not repo_ckpt.exists():
            repo_ckpt.symlink_to(sm_ckpt)
            print(f"Linked checkpoints: {sm_ckpt} -> {repo_ckpt}")

    # Outputs
    repo_outputs = repo / "outputs"
    repo_outputs.mkdir(parents=True, exist_ok=True)

    return repo


def run_script(script_name: str, args: list[str] | None = None):
    """Run a pipeline script from the repo's scripts/ directory."""
    script = REPO_ROOT / "scripts" / script_name
    cmd = [sys.executable, str(script)] + (args or [])
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"FAILED: {script_name} (exit code {result.returncode})")
        sys.exit(result.returncode)


def copy_outputs_to_model_dir():
    """Copy final outputs to SM_MODEL_DIR for S3 upload."""
    outputs = REPO_ROOT / "outputs"
    if outputs.exists():
        dst = Path(SM_MODEL_DIR) / "outputs"
        if outputs.is_symlink():
            # Copy the actual content, not the symlink
            shutil.copytree(str(outputs.resolve()), str(dst), dirs_exist_ok=True)
        else:
            shutil.copytree(str(outputs), str(dst), dirs_exist_ok=True)
        print(f"Outputs copied to {dst}")


def main():
    parser = argparse.ArgumentParser(description="SageMaker training entry point")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["data", "task-tune", "phase1", "phase2", "phase3", "all"],
        help="Which phase to run",
    )
    parser.add_argument(
        "--task",
        default="qd",
        choices=["qd", "finqa"],
        help="Task for task-tuning phase",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data pipeline (assume data already exists in channel)",
    )
    args = parser.parse_args()

    repo = setup_paths()
    os.chdir(str(repo))

    # Install the package
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=str(repo))

    if args.phase == "data":
        run_script("01_download_data.py")
        run_script("02_build_corpus.py")
        run_script("03_build_faiss_index.py")
        run_script("04_extract_triples.py")
        run_script("05_generate_qd_data.py")
        run_script("06_build_locality_facts.py")

    elif args.phase == "task-tune":
        if args.task == "qd":
            run_script("07_task_tune_qd.py")
        else:
            run_script("08_task_tune_finqa.py")

    elif args.phase == "phase1":
        run_script("11_run_phase1.py")

    elif args.phase == "phase2":
        run_script("12_run_phase2.py")

    elif args.phase == "phase3":
        run_script("13_run_phase3.py")

    elif args.phase == "all":
        if not args.skip_data:
            run_script("01_download_data.py")
            run_script("02_build_corpus.py")
            run_script("03_build_faiss_index.py")
            run_script("04_extract_triples.py")
            run_script("05_generate_qd_data.py")
            run_script("06_build_locality_facts.py")

        run_script("07_task_tune_qd.py")
        run_script("11_run_phase1.py")
        run_script("12_run_phase2.py")

        # Phase 3 requires FinQA tuning
        run_script("08_task_tune_finqa.py")
        run_script("13_run_phase3.py")

        run_script("14_generate_tables.py")

    copy_outputs_to_model_dir()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
