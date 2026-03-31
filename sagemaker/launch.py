"""Launch a SageMaker training job.

Usage:
    # First time setup:
    export WANDB_API_KEY=your-key
    export HF_TOKEN=your-token          # optional, Qwen2.5-3B is public
    export SAGEMAKER_ROLE=arn:aws:iam::...  # or auto-detected in notebook

    # Upload data once (after running pipeline locally or on Colab):
    python sagemaker/sync.py upload

    # Launch experiments:
    python sagemaker/launch.py --phase phase1 --instance ml.g5.2xlarge --spot
    python sagemaker/launch.py --phase all --instance ml.g5.2xlarge --spot --skip-data

    # Download results:
    python sagemaker/sync.py download

Requires:
    pip install sagemaker boto3
"""

import argparse  # noqa: I001
import os

import sagemaker
from sagemaker.huggingface import HuggingFace


S3_PREFIX = "still-on-task"


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["data", "task-tune", "phase1", "phase2", "phase3", "all"],
    )
    parser.add_argument("--task", default="qd", choices=["qd", "finqa"])
    parser.add_argument("--instance", default="ml.g5.2xlarge", help="SageMaker instance type")
    parser.add_argument("--spot", action="store_true", help="Use spot instances (cheaper)")
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket (default: SageMaker default)")
    parser.add_argument("--s3-data", default=None, help="S3 URI for data channel (auto if omitted)")
    parser.add_argument("--s3-output", default=None, help="S3 URI for output")
    parser.add_argument("--role", default=None, help="SageMaker execution role ARN")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--max-run", type=int, default=86400, help="Max run time in seconds")
    parser.add_argument("--job-name", default=None, help="Custom job name")
    args = parser.parse_args()

    session = sagemaker.Session()
    role = args.role or os.environ.get("SAGEMAKER_ROLE") or sagemaker.get_execution_role()
    bucket = args.s3_bucket or session.default_bucket()

    s3_output = args.s3_output or f"s3://{bucket}/{S3_PREFIX}/output"
    s3_checkpoint = f"s3://{bucket}/{S3_PREFIX}/checkpoints"

    # Pass environment variables from local env to SageMaker container
    env = {}
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key
        print("WANDB_API_KEY found, will log to WandB")
    else:
        env["WANDB_DISABLED"] = "true"
        print("No WANDB_API_KEY set, WandB disabled")
    if hf_token:
        env["HF_TOKEN"] = hf_token

    hyperparameters = {
        "phase": args.phase,
        "task": args.task,
    }
    if args.skip_data:
        hyperparameters["skip-data"] = ""

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir="sagemaker",
        instance_type=args.instance,
        instance_count=1,
        role=role,
        transformers_version="4.44.0",
        pytorch_version="2.3.0",
        py_version="py311",
        hyperparameters=hyperparameters,
        output_path=s3_output,
        max_run=args.max_run,
        use_spot_instances=args.spot,
        max_wait=args.max_run * 2 if args.spot else None,
        checkpoint_s3_uri=s3_checkpoint if args.spot else None,
        environment=env,
        base_job_name=args.job_name or f"sot-{args.phase}",
    )

    # Data channel — auto-detect from S3 if not specified
    inputs = {}
    if args.s3_data:
        inputs["data"] = args.s3_data
    else:
        default_data = f"s3://{bucket}/{S3_PREFIX}/data"
        # Check if data exists at the default location
        import boto3

        s3 = boto3.client("s3")
        try:
            s3.head_object(
                Bucket=bucket, Key=f"{S3_PREFIX}/data/fnspid/processed/pre_cutoff.parquet"
            )
            inputs["data"] = default_data
            print(f"Found data at {default_data}")
        except s3.exceptions.ClientError:
            if args.phase not in ("data", "all"):
                print(f"WARNING: No data found at {default_data}")
                print("  Run 'python sagemaker/sync.py upload' first, or use --phase data")

    print("\nLaunching SageMaker job:")
    print(f"  Phase:    {args.phase}")
    print(f"  Instance: {args.instance}")
    print(f"  Spot:     {args.spot}")
    print(f"  Output:   {s3_output}")
    if inputs:
        print(f"  Data:     {inputs.get('data', 'downloading at runtime')}")
    print()

    estimator.fit(inputs if inputs else None, job_name=args.job_name)

    print("\nJob complete.")
    print(f"  Output:  {s3_output}")
    print(
        f"  Download: python sagemaker/sync.py download --job {estimator.latest_training_job.name}"
    )


if __name__ == "__main__":
    main()
