"""Launch a SageMaker training job.

Usage:
    python sagemaker/launch.py --phase phase1 --instance ml.g5.2xlarge
    python sagemaker/launch.py --phase all --instance ml.p3.2xlarge --spot

Requires:
    - AWS credentials configured (aws configure)
    - sagemaker SDK: pip install sagemaker
    - An S3 bucket for data/outputs
"""

import argparse  # noqa: I001

import sagemaker
from sagemaker.huggingface import HuggingFace


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
    parser.add_argument("--s3-data", default=None, help="S3 URI to pre-uploaded data channel")
    parser.add_argument("--s3-output", default=None, help="S3 URI for output")
    parser.add_argument("--role", default=None, help="SageMaker execution role ARN")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--max-run", type=int, default=86400, help="Max run time in seconds")
    args = parser.parse_args()

    session = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role()
    default_bucket = session.default_bucket()

    s3_output = args.s3_output or f"s3://{default_bucket}/still-on-task/output"

    hyperparameters = {
        "phase": args.phase,
        "task": args.task,
    }
    if args.skip_data:
        hyperparameters["skip-data"] = ""

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir="sagemaker",
        git_config=None,  # Uses local source_dir
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
        checkpoint_s3_uri=f"s3://{default_bucket}/still-on-task/checkpoints" if args.spot else None,
        environment={
            "WANDB_API_KEY": "",  # Set via env or .env
            "HF_TOKEN": "",  # Set if model is gated
        },
        dependencies=["pyproject.toml"],
    )

    # Data channels
    inputs = {}
    if args.s3_data:
        inputs["data"] = args.s3_data

    print("Launching SageMaker job:")
    print(f"  Phase: {args.phase}")
    print(f"  Instance: {args.instance}")
    print(f"  Spot: {args.spot}")
    print(f"  Output: {s3_output}")

    estimator.fit(inputs if inputs else None)
    print(f"\nJob complete. Output at: {s3_output}")


if __name__ == "__main__":
    main()
