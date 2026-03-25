"""
SageMaker job launcher for FullVideoClassifier v2 training.

Prerequisites:
    pip install sagemaker boto3
    AWS credentials configured (aws configure or IAM role)

Usage:
    python launch_sagemaker.py                    # spot instance (recommended)
    python launch_sagemaker.py --no-spot          # on-demand (more reliable, ~3x cost)
    python launch_sagemaker.py --dry-run          # print config without launching

Cost reference (us-west-2, approximate):
    ml.g5.2xlarge spot:      ~$0.55/hr  →  24hr ≈ $13
    ml.g5.2xlarge on-demand: ~$1.52/hr  →  24hr ≈ $36
"""

import argparse

# ---------------------------------------------------------------------------
# Configuration — edit these before launching
# ---------------------------------------------------------------------------

BUCKET          = 'genvideo-complete'
DATA_PREFIX     = 'complete_dataset/dataset/'   # S3 prefix for training videos
CHECKPOINT_PREFIX = 'checkpoints/fullvideo_v2/' # S3 prefix for spot checkpoints
OUTPUT_PREFIX   = 'output/fullvideo_v2/'        # S3 prefix for final model
REGION          = 'us-west-2'

INSTANCE_TYPE   = 'ml.g5.2xlarge'   # 1x A10G 24GB VRAM, 8 vCPU
PYTORCH_VERSION = '2.8.0'
PYTHON_VERSION  = 'py312'

HYPERPARAMETERS = {
    'epochs':          15,
    'batch-size':      8,     # 8 × 24frames × 512px fits in 24GB with AMP
    'learning-rate':   1e-5,
    'warmup-epochs':   2,
    'target-size':     512,
    'max-frames':      24,
    'num-workers':     4,
    'label-smoothing': 0.05,
    'seed':            314159,
}

MAX_RUN_SECONDS  = 90_000   # 25 hours — enough buffer beyond expected 24hr
MAX_WAIT_SECONDS = 7_200    # 2 hours to wait for spot capacity before failing

# ---------------------------------------------------------------------------

def get_role():
    """Get SageMaker execution role. Tries session role first, then prompts."""
    import sagemaker
    try:
        return sagemaker.get_execution_role()
    except Exception:
        import os
        role = os.environ.get('SAGEMAKER_ROLE')
        if role:
            return role
        raise RuntimeError(
            "Could not determine SageMaker role. Set env var SAGEMAKER_ROLE "
            "or run from a SageMaker notebook/instance."
        )


def build_estimator(role, use_spot, job_name):
    from sagemaker.pytorch import PyTorch
    s3_checkpoint_uri = f's3://{BUCKET}/{CHECKPOINT_PREFIX}'
    s3_output_path    = f's3://{BUCKET}/{OUTPUT_PREFIX}'

    kwargs = dict(
        entry_point        = 'sm_train_v2.py',
        source_dir         = '.',                     # uploads entire project dir
        framework_version  = PYTORCH_VERSION,
        py_version         = PYTHON_VERSION,
        instance_type      = INSTANCE_TYPE,
        instance_count     = 1,
        hyperparameters    = HYPERPARAMETERS,
        output_path        = s3_output_path,
        role               = role,
        max_run            = MAX_RUN_SECONDS,
        # SageMaker reads requirements.txt from source_dir automatically
        # We point it to the SageMaker-specific one via env var trick below
        environment        = {
            'PIP_REQUIREMENTS': 'sagemaker_requirements.txt',
        },
    )

    if use_spot:
        kwargs.update(
            use_spot_instances   = True,
            checkpoint_s3_uri    = s3_checkpoint_uri,
            checkpoint_local_path= '/opt/ml/checkpoints',
            max_wait             = MAX_WAIT_SECONDS,
        )
    else:
        # On-demand: still checkpoint for safety
        kwargs['checkpoint_s3_uri']     = s3_checkpoint_uri
        kwargs['checkpoint_local_path'] = '/opt/ml/checkpoints'

    estimator = PyTorch(**kwargs)
    return estimator, s3_output_path


def estimate_cost(use_spot, max_run_hours=25):
    rates = {
        'ml.g5.2xlarge': {'spot': 0.55, 'on_demand': 1.515},
    }
    r = rates.get(INSTANCE_TYPE, {'spot': 1.0, 'on_demand': 2.0})
    rate = r['spot'] if use_spot else r['on_demand']
    low  = rate * 20   # optimistic: 20hr
    high = rate * max_run_hours
    return low, high, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-spot', action='store_true',
                        help='Use on-demand instead of spot instances')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configuration without launching')
    parser.add_argument('--job-name', type=str, default=None,
                        help='Override job name (auto-generated if omitted)')
    args = parser.parse_args()

    use_spot = not args.no_spot

    from datetime import datetime
    job_name = args.job_name or f'fullvideo-v2-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    low, high, rate = estimate_cost(use_spot)

    print("\n" + "="*60)
    print("SAGEMAKER JOB CONFIGURATION")
    print("="*60)
    print(f"  Job name:      {job_name}")
    print(f"  Instance:      {INSTANCE_TYPE}  ({'SPOT' if use_spot else 'ON-DEMAND'})")
    print(f"  Framework:     PyTorch {PYTORCH_VERSION} / {PYTHON_VERSION}")
    print(f"  Data:          s3://{BUCKET}/{DATA_PREFIX}")
    print(f"  Checkpoints:   s3://{BUCKET}/{CHECKPOINT_PREFIX}")
    print(f"  Output:        s3://{BUCKET}/{OUTPUT_PREFIX}")
    print(f"  Rate:          ~${rate:.2f}/hr")
    print(f"  Cost estimate: ${low:.0f}–${high:.0f}  (20–25hr window)")
    print(f"  Max wait:      {MAX_WAIT_SECONDS//3600}hr (spot only)")
    print(f"  Max run:       {MAX_RUN_SECONDS//3600}hr")
    print("\nHyperparameters:")
    for k, v in HYPERPARAMETERS.items():
        print(f"  {k}: {v}")
    print("="*60)

    if args.dry_run:
        print("\nDry run — not launching.")
        return

    # Validate dependencies before asking for confirmation
    try:
        import boto3
        import sagemaker
        from sagemaker.pytorch import PyTorch
        from sagemaker.inputs import TrainingInput
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Run: pip install sagemaker boto3")
        return

    # Confirm before spending money
    confirm = input("\nLaunch job? [y/N] ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    boto_session = boto3.Session(region_name=REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)  # noqa: F841
    role = get_role()

    estimator, output_path = build_estimator(role, use_spot, job_name)

    # File mode: SageMaker copies 16GB dataset to EBS (~2-3 min), then reads locally
    training_input = TrainingInput(
        s3_data    = f's3://{BUCKET}/{DATA_PREFIX}',
        input_mode = 'File',
    )

    print(f"\nSubmitting job: {job_name}")
    estimator.fit(
        inputs   = {'training': training_input},
        job_name = job_name,
        wait     = False,   # Don't block — monitor via CloudWatch or console
    )

    print(f"\nJob submitted: {job_name}")
    print(f"Monitor: https://console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{job_name}")
    print(f"Logs:    https://console.aws.amazon.com/cloudwatch/home?region={REGION}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={job_name}")
    print(f"Output will be at: {output_path}")


if __name__ == '__main__':
    main()
