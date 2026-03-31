"""Upload data to S3 and download results from SageMaker jobs.

Usage:
    # Upload local data/checkpoints to S3 for SageMaker
    python sagemaker/sync.py upload

    # Upload only specific directories
    python sagemaker/sync.py upload --dirs data checkpoints

    # Download results from the latest job
    python sagemaker/sync.py download

    # Download results from a specific job
    python sagemaker/sync.py download --job sot-phase1-2026-03-31-12-00-00

    # List recent training jobs
    python sagemaker/sync.py jobs
"""

import argparse  # noqa: I001
import subprocess
import sys
import tarfile
from pathlib import Path

import boto3

S3_PREFIX = "still-on-task"
REPO_ROOT = Path(__file__).resolve().parents[1]


def get_bucket():
    """Get the default SageMaker bucket."""
    import sagemaker

    return sagemaker.Session().default_bucket()


def upload(bucket: str, dirs: list[str] | None = None):
    """Upload local data and checkpoints to S3."""
    default_dirs = ["data", "checkpoints"]
    dirs_to_upload = dirs or default_dirs

    for d in dirs_to_upload:
        local_path = REPO_ROOT / d
        if not local_path.exists():
            print(f"Skipping {d}/ (does not exist)")
            continue

        s3_uri = f"s3://{bucket}/{S3_PREFIX}/{d}/"
        print(f"\nUploading {local_path} -> {s3_uri}")

        # Use aws cli for fast parallel upload
        cmd = [
            "aws",
            "s3",
            "sync",
            str(local_path),
            s3_uri,
            "--exclude",
            "*.pyc",
            "--exclude",
            "__pycache__/*",
            "--exclude",
            ".cache/*",
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: upload {d}/")
            sys.exit(1)

        print(f"Done: {d}/")

    print("\nAll uploads complete.")
    print("Use this as your data channel:")
    print(f"  --s3-data s3://{bucket}/{S3_PREFIX}/data/")


def download(bucket: str, job_name: str | None = None, output_dir: str = "sagemaker-results"):
    """Download results from a SageMaker job."""
    sm = boto3.client("sagemaker")
    s3 = boto3.client("s3")

    # Find the job
    if job_name:
        job = sm.describe_training_job(TrainingJobName=job_name)
    else:
        # Get most recent job
        jobs = sm.list_training_jobs(
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
            NameContains="sot-",
        )
        if not jobs["TrainingJobSummaries"]:
            print("No SageMaker jobs found with prefix 'sot-'")
            sys.exit(1)
        job_name = jobs["TrainingJobSummaries"][0]["TrainingJobName"]
        job = sm.describe_training_job(TrainingJobName=job_name)

    status = job["TrainingJobStatus"]
    print(f"Job: {job_name}")
    print(f"Status: {status}")

    if status != "Completed":
        if status == "InProgress":
            print("Job still running. Check WandB for live metrics or wait.")
        else:
            print(f"Job {status}. Check CloudWatch logs:")
            print(
                f"  aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs "
                f"--log-stream-name {job_name}/algo-1-stdout"
            )
        return

    # Download model.tar.gz (contains outputs/)
    model_s3 = job["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"\nDownloading from: {model_s3}")

    local_dir = REPO_ROOT / output_dir / job_name
    local_dir.mkdir(parents=True, exist_ok=True)
    tar_path = local_dir / "model.tar.gz"

    # Parse S3 URI
    parts = model_s3.replace("s3://", "").split("/", 1)
    s3_bucket = parts[0]
    s3_key = parts[1]

    s3.download_file(s3_bucket, s3_key, str(tar_path))
    print(f"Downloaded: {tar_path}")

    # Extract
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=str(local_dir))
    tar_path.unlink()

    # Copy results into repo outputs/ for local analysis
    extracted_outputs = local_dir / "outputs"
    repo_outputs = REPO_ROOT / "outputs"
    if extracted_outputs.exists():
        print(f"\nCopying results to {repo_outputs}")
        subprocess.run(["cp", "-r", str(extracted_outputs) + "/.", str(repo_outputs)])
        print("Run 'python scripts/14_generate_tables.py' to generate result tables.")

    # Show what we got
    print(f"\nResults at: {local_dir}")
    for f in sorted(local_dir.rglob("eval_results.json")):
        print(f"  {f.relative_to(local_dir)}")


def list_jobs(bucket: str):
    """List recent SageMaker training jobs."""
    sm = boto3.client("sagemaker")
    jobs = sm.list_training_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=20,
        NameContains="sot-",
    )

    if not jobs["TrainingJobSummaries"]:
        print("No SageMaker jobs found with prefix 'sot-'")
        return

    print(f"{'Job Name':<50} {'Status':<12} {'Created'}")
    print("-" * 90)
    for j in jobs["TrainingJobSummaries"]:
        name = j["TrainingJobName"]
        status = j["TrainingJobStatus"]
        created = j["CreationTime"].strftime("%Y-%m-%d %H:%M")
        print(f"{name:<50} {status:<12} {created}")


def main():
    parser = argparse.ArgumentParser(description="Sync data/results with S3")
    parser.add_argument("action", choices=["upload", "download", "jobs"])
    parser.add_argument("--bucket", default=None, help="S3 bucket")
    parser.add_argument("--job", default=None, help="Job name for download")
    parser.add_argument("--dirs", nargs="*", help="Directories to upload")
    parser.add_argument("--output-dir", default="sagemaker-results", help="Local dir for downloads")
    args = parser.parse_args()

    bucket = args.bucket or get_bucket()

    if args.action == "upload":
        upload(bucket, args.dirs)
    elif args.action == "download":
        download(bucket, args.job, args.output_dir)
    elif args.action == "jobs":
        list_jobs(bucket)


if __name__ == "__main__":
    main()
