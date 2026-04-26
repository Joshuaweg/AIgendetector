"""
SageMaker job launcher for Stage 2 optical flow fusion training (sm_train_v3.py).

Uploads backbone.pt + flow_stage1.pt to S3 as a 'checkpoints' input channel,
then launches the training job with those weights pre-loaded.

Prerequisites:
    pip install sagemaker boto3
    AWS credentials configured (aws configure or IAM role)

Usage:
    python launch_flow_sagemaker.py                    # spot instance (recommended)
    python launch_flow_sagemaker.py --no-spot          # on-demand (~3x cost)
    python launch_flow_sagemaker.py --dry-run          # print config without launching

Cost reference (us-west-2, approximate):
    ml.g5.2xlarge spot:      ~$0.55/hr  →  16hr ≈ $9
    ml.g5.2xlarge on-demand: ~$1.52/hr  →  16hr ≈ $24
"""

import argparse
import os

# ---------------------------------------------------------------------------
# Configuration — edit before launching
# ---------------------------------------------------------------------------

BUCKET            = 'genvideo-complete'
DATA_PREFIX       = 's2_flow_dataset/'                 # S3 prefix for training videos
CHECKPOINT_PREFIX = 'checkpoints/flow_stage2/'        # S3 prefix for spot checkpoints
INIT_PREFIX       = 'checkpoints/flow_stage2_init/'   # S3 prefix for backbone + flow init weights
OUTPUT_PREFIX     = 'output/flow_stage2/'             # S3 prefix for final model
REGION            = 'us-west-2'

INSTANCE_TYPE   = 'ml.g5.2xlarge'   # 1x A10G 24GB VRAM, 8 vCPU
PYTORCH_VERSION = '2.8.0'
PYTHON_VERSION  = 'py312'

# Local paths to upload as init weights
LOCAL_BACKBONE_CHECKPOINT  = 'model/checkpoint_epoch_0004.pt'   # 92% backbone
LOCAL_FLOW_CHECKPOINT      = 'flow_model_output/best_flow_model.pt'  # Stage 1 FlowEncoder

HYPERPARAMETERS = {
    'epochs':          15,
    'batch-size':      8,      # 8 × 24frames × 512px fits in 24GB with AMP
    'learning-rate':   1e-4,   # FlowEncoder + Classifier only (backbone frozen)
    'warmup-epochs':   2,
    'target-size':     512,
    'max-frames':      24,
    'flow-h':          64,
    'flow-w':          64,
    'num-workers':     8,
    'label-smoothing': 0.05,
    'seed':            314159,
}

MAX_RUN_SECONDS  = 72_000    # 20 hours
MAX_WAIT_SECONDS = 79_200    # MAX_RUN + 2hr buffer (must be >= MAX_RUN for spot)

# ---------------------------------------------------------------------------

def get_role():
    import sagemaker
    try:
        return sagemaker.get_execution_role()
    except Exception:
        role = os.environ.get('SAGEMAKER_ROLE')
        if role:
            return role
        raise RuntimeError(
            "Could not determine SageMaker role. Set env var SAGEMAKER_ROLE "
            "or run from a SageMaker notebook/instance."
        )


def upload_init_weights(s3_client, dry_run=False):
    """Upload backbone.pt + flow_stage1.pt to S3 INIT_PREFIX. Returns S3 URI."""
    uploads = [
        (LOCAL_BACKBONE_CHECKPOINT, 'backbone.pt'),
        (LOCAL_FLOW_CHECKPOINT,     'flow_stage1.pt'),
    ]
    for local_path, s3_name in uploads:
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Init weight not found locally: {local_path}\n"
                f"Run Stage 1 training first or check the path."
            )
        s3_key = f"{INIT_PREFIX}{s3_name}"
        print(f"  {'[DRY RUN] Would upload' if dry_run else 'Uploading'} "
              f"{local_path} → s3://{BUCKET}/{s3_key}")
        if not dry_run:
            s3_client.upload_file(local_path, BUCKET, s3_key)

    return f's3://{BUCKET}/{INIT_PREFIX}'


def build_estimator(role, use_spot, job_name, checkpoints_s3_uri, sagemaker_session=None):
    from sagemaker.pytorch import PyTorch
    s3_checkpoint_uri = f's3://{BUCKET}/{CHECKPOINT_PREFIX}'
    s3_output_path    = f's3://{BUCKET}/{OUTPUT_PREFIX}'

    kwargs = dict(
        entry_point        = 'sm_train_v3.py',
        source_dir         = '.',
        framework_version  = PYTORCH_VERSION,
        py_version         = PYTHON_VERSION,
        instance_type      = INSTANCE_TYPE,
        instance_count     = 1,
        hyperparameters    = HYPERPARAMETERS,
        output_path        = s3_output_path,
        role               = role,
        max_run            = MAX_RUN_SECONDS,
        sagemaker_session  = sagemaker_session,
        environment        = {
            'PIP_REQUIREMENTS': 'sagemaker_requirements.txt',
        },
    )

    if use_spot:
        kwargs.update(
            use_spot_instances    = True,
            checkpoint_s3_uri     = s3_checkpoint_uri,
            checkpoint_local_path = '/opt/ml/checkpoints',
            max_wait              = MAX_WAIT_SECONDS,
        )
    else:
        kwargs['checkpoint_s3_uri']     = s3_checkpoint_uri
        kwargs['checkpoint_local_path'] = '/opt/ml/checkpoints'

    estimator = PyTorch(**kwargs)
    return estimator, s3_output_path


def estimate_cost(use_spot, max_run_hours=20):
    rates = {'ml.g5.2xlarge': {'spot': 0.55, 'on_demand': 1.515}}
    r    = rates.get(INSTANCE_TYPE, {'spot': 1.0, 'on_demand': 2.0})
    rate = r['spot'] if use_spot else r['on_demand']
    return rate * 10, rate * max_run_hours, rate   # low, high, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-spot',  action='store_true', help='Use on-demand instead of spot')
    parser.add_argument('--dry-run',  action='store_true', help='Print config without launching')
    parser.add_argument('--job-name', type=str, default=None, help='Override auto-generated job name')
    args = parser.parse_args()

    use_spot = not args.no_spot

    from datetime import datetime
    job_name = args.job_name or f'flow-stage2-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    low, high, rate = estimate_cost(use_spot)

    print("\n" + "="*60)
    print("SAGEMAKER JOB CONFIGURATION — Stage 2 Flow Fusion")
    print("="*60)
    print(f"  Job name:      {job_name}")
    print(f"  Entry point:   sm_train_v3.py")
    print(f"  Instance:      {INSTANCE_TYPE}  ({'SPOT' if use_spot else 'ON-DEMAND'})")
    print(f"  Framework:     PyTorch {PYTORCH_VERSION} / {PYTHON_VERSION}")
    print(f"  Data:          s3://{BUCKET}/{DATA_PREFIX}")
    print(f"  Init weights:  s3://{BUCKET}/{INIT_PREFIX}")
    print(f"    backbone.pt  ← {LOCAL_BACKBONE_CHECKPOINT}")
    print(f"    flow_stage1.pt ← {LOCAL_FLOW_CHECKPOINT}")
    print(f"  Checkpoints:   s3://{BUCKET}/{CHECKPOINT_PREFIX}")
    print(f"  Output:        s3://{BUCKET}/{OUTPUT_PREFIX}")
    print(f"  Rate:          ~${rate:.2f}/hr")
    print(f"  Cost estimate: ${low:.0f}–${high:.0f}  (10–20hr window)")
    print(f"  Max wait:      {MAX_WAIT_SECONDS//3600}hr (spot only)")
    print(f"  Max run:       {MAX_RUN_SECONDS//3600}hr")
    print(f"\nFreezing: LatentEncoder + PatchEncoder")
    print(f"Training: FlowEncoder + Classifier")
    print(f"\nHyperparameters:")
    for k, v in HYPERPARAMETERS.items():
        print(f"  {k}: {v}")
    print("="*60)

    if args.dry_run:
        print("\nDry run — not launching.")
        # Still validate local files exist
        for path, name in [(LOCAL_BACKBONE_CHECKPOINT, 'backbone'), (LOCAL_FLOW_CHECKPOINT, 'flow stage 1')]:
            exists = os.path.exists(path)
            print(f"  {'✓' if exists else '✗'} {name}: {path}")
        return

    try:
        import boto3
        import sagemaker
        from sagemaker.pytorch import PyTorch
        from sagemaker.inputs import TrainingInput
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Run: pip install sagemaker boto3")
        return

    confirm = input("\nLaunch job? [y/N] ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    boto_session      = boto3.Session(region_name=REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    s3_client         = boto_session.client('s3')
    role              = get_role()

    # Upload init weights to S3
    print("\nUploading init weights to S3...")
    checkpoints_s3_uri = upload_init_weights(s3_client, dry_run=False)
    print(f"Init weights at: {checkpoints_s3_uri}")

    estimator, output_path = build_estimator(
        role, use_spot, job_name, checkpoints_s3_uri, sagemaker_session
    )

    training_input = TrainingInput(
        s3_data    = f's3://{BUCKET}/{DATA_PREFIX}',
        input_mode = 'File',
    )
    checkpoints_input = TrainingInput(
        s3_data    = checkpoints_s3_uri,
        input_mode = 'File',
    )

    print(f"\nSubmitting job: {job_name}")
    estimator.fit(
        inputs   = {'training': training_input, 'checkpoints': checkpoints_input},
        job_name = job_name,
        wait     = False,
    )

    print(f"\nJob submitted: {job_name}")
    print(f"Monitor: https://console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{job_name}")
    print(f"Logs:    https://console.aws.amazon.com/cloudwatch/home?region={REGION}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={job_name}")
    print(f"Output will be at: {output_path}")


if __name__ == '__main__':
    main()
