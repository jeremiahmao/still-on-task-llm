"""SageMaker training entry point.

Usage on SageMaker:
    The estimator calls this script with --phase and optional --methods/--scale args.
    Data is expected at /opt/ml/input/data/data/ (SageMaker channel)
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
    elif not repo_data.exists():
        repo_data.mkdir(parents=True, exist_ok=True)
        print(f"Created empty data dir: {repo_data}")

    # Checkpoints: use SageMaker checkpoint dir for spot instance resilience
    sm_ckpt = Path(SM_CHECKPOINT_DIR)
    repo_ckpt = repo / "checkpoints"
    if sm_ckpt.exists():
        sm_ckpt.mkdir(parents=True, exist_ok=True)
        if not repo_ckpt.exists():
            repo_ckpt.symlink_to(sm_ckpt)
            print(f"Linked checkpoints: {sm_ckpt} -> {repo_ckpt}")
    elif not repo_ckpt.exists():
        repo_ckpt.mkdir(parents=True, exist_ok=True)

    # Outputs
    repo_outputs = repo / "outputs"
    repo_outputs.mkdir(parents=True, exist_ok=True)

    return repo


def _build_cmd(script_path: str, args: list[str], distributed: bool) -> list[str]:
    nproc = int(os.environ.get("NPROC_PER_NODE", "1"))
    if distributed and nproc > 1:
        return ["torchrun", f"--nproc_per_node={nproc}", script_path] + args
    return [sys.executable, script_path] + args


def run_script(script_name: str, args: list[str] | None = None, distributed: bool = False):
    """Run a pipeline script from the repo's scripts/ directory."""
    script = REPO_ROOT / "scripts" / script_name
    cmd = _build_cmd(str(script), args or [], distributed)
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"FAILED: {script_name} (exit code {result.returncode})")
        sys.exit(result.returncode)


def copy_to_model_dir():
    """Copy outputs and key artifacts to SM_MODEL_DIR for S3 upload."""
    model_dir = Path(SM_MODEL_DIR)
    if not model_dir.exists():
        print(f"Skipping model dir copy (not a SageMaker training job)")
        return

    # Copy outputs/ (eval results, metadata, configs)
    outputs = REPO_ROOT / "outputs"
    if outputs.exists():
        dst = model_dir / "outputs"
        if outputs.is_symlink():
            shutil.copytree(str(outputs.resolve()), str(dst), dirs_exist_ok=True)
        else:
            shutil.copytree(str(outputs), str(dst), dirs_exist_ok=True)
        print(f"Outputs -> {dst}")

    # Copy generated data artifacts (triples, QD data, locality facts)
    # so they can be reused without re-running the data pipeline
    data_artifacts = [
        "data/fnspid/triples",
        "data/fnspid/locality_facts.json",
        "data/qd",
    ]
    for artifact in data_artifacts:
        src = REPO_ROOT / artifact
        if src.exists():
            dst = model_dir / artifact
            if src.is_dir():
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
            print(f"{artifact} -> {dst}")

    # Copy the results CSV if it was generated
    csv_path = REPO_ROOT / "outputs" / "all_results.csv"
    if csv_path.exists():
        shutil.copy2(str(csv_path), str(model_dir / "all_results.csv"))


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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run with debug ticker subset for fast smoke test",
    )
    parser.add_argument(
        "--teacher-provider",
        default="cerebras",
        choices=["local", "openai", "cerebras"],
        help="Teacher LLM provider for data generation (default: cerebras)",
    )
    args = parser.parse_args()

    repo = setup_paths()
    os.chdir(str(repo))

    # Install the package
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=str(repo))

    # Common flags for data pipeline scripts
    debug_flag = ["--debug"] if args.debug else []
    teacher_flags = ["--provider", args.teacher_provider]
    teacher_flags_qd = ["--teacher-provider", args.teacher_provider]

    if args.phase == "data":
        run_script("01_download_data.py", ["--skip-finqa"] if args.debug else [])
        run_script("02_build_corpus.py", debug_flag)
        run_script("03_build_faiss_index.py", debug_flag)
        run_script("04_extract_triples.py", debug_flag + teacher_flags)
        run_script("05_generate_qd_data_foundational_model.py", debug_flag + teacher_flags_qd)
        run_script("06_build_locality_facts.py", debug_flag)

    elif args.phase == "task-tune":
        if args.task == "qd":
            run_script("07_task_tune_qd.py", debug_flag, distributed=True)
        else:
            run_script("08_task_tune_finqa.py", distributed=True)

    elif args.phase == "phase1":
        run_script("11_run_phase1.py", debug_flag)

    elif args.phase == "phase2":
        run_script("12_run_phase2.py", debug_flag)

    elif args.phase == "phase3":
        run_script("13_run_phase3.py", debug_flag)

    elif args.phase == "all":
        if not args.skip_data:
            run_script("01_download_data.py", ["--skip-finqa"] if args.debug else [])
            run_script("02_build_corpus.py", debug_flag)
            run_script("03_build_faiss_index.py", debug_flag)
            run_script("04_extract_triples.py", debug_flag + teacher_flags)
            run_script("05_generate_qd_data_foundational_model.py", debug_flag + teacher_flags_qd)
            run_script("06_build_locality_facts.py", debug_flag)

        run_script("07_task_tune_qd.py", debug_flag, distributed=True)
        run_script("11_run_phase1.py", debug_flag)
        run_script("12_run_phase2.py", debug_flag)

        # Phase 3 requires FinQA tuning — skip in debug (FinQA not downloaded)
        if not args.debug:
            run_script("08_task_tune_finqa.py", distributed=True)
            run_script("13_run_phase3.py", debug_flag)

        run_script("14_generate_tables.py")

    copy_to_model_dir()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
