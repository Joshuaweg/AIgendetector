"""
Direct boto3 launcher for Stage 2 flow fusion training.
Bypasses the SageMaker Python SDK entirely — calls create_training_job directly.
Exits immediately after job submission (no hanging).

Usage:
    python launch_boto3.py
    python launch_boto3.py --no-spot
    python launch_boto3.py --dry-run
"""

import argparse
import json
import os
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Configuration — must match launch_flow_sagemaker.py
# ---------------------------------------------------------------------------
BUCKET             = 'genvideo-complete'
DATA_PREFIX        = 's2_flow_dataset/'
FLOW_CACHE_PREFIX  = 's2_flow_cache/'
CHECKPOINT_PREFIX  = 'checkpoints/flow_stage2/'
INIT_PREFIX        = 'checkpoints/flow_stage2_init/'
OUTPUT_PREFIX      = 'output/flow_stage2/'
REGION             = 'us-west-2'

ROLE_ARN        = os.environ.get('SAGEMAKER_ROLE', 'arn:aws:iam::507463957484:role/aigendectector-role')
INSTANCE_TYPE   = 'ml.g5.2xlarge'
PYTORCH_VERSION = '2.8.0'
PYTHON_VERSION  = 'py312'

LOCAL_BACKBONE_CHECKPOINT = 'model/checkpoint_epoch_0004.pt'
LOCAL_FLOW_CHECKPOINT     = 'flow_model_output/best_flow_model.pt'

HYPERPARAMETERS = {
    'epochs':          '15',
    'batch-size':      '8',
    'learning-rate':   '0.0001',
    'warmup-epochs':   '2',
    'target-size':     '512',
    'max-frames':      '24',
    'flow-h':          '64',
    'flow-w':          '64',
    'num-workers':     '8',
    'label-smoothing': '0.05',
    'seed':            '314159',
}

MAX_RUN_SECONDS  = 72_000
MAX_WAIT_SECONDS = 79_200


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no-spot',           action='store_true')
    p.add_argument('--dry-run',           action='store_true')
    p.add_argument('--job-name',          type=str, default=None)
    p.add_argument('--force-init-upload', action='store_true',
                   help='Re-upload init weights even if already present on S3')
    return p.parse_args()


def s3_key_exists(s3_client, key):
    try:
        s3_client.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError:
        return False


def upload_init_weights(s3_client, dry_run=False, force=False):
    uploads = [
        (LOCAL_BACKBONE_CHECKPOINT, 'backbone.pt'),
        (LOCAL_FLOW_CHECKPOINT,     'flow_stage1.pt'),
    ]
    for local_path, s3_name in uploads:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Not found: {local_path}")
        s3_key = f"{INIT_PREFIX}{s3_name}"
        if not force and not dry_run and s3_key_exists(s3_client, s3_key):
            print(f"  Skipping {s3_name} — already on S3 (use --force-init-upload to overwrite)")
            continue
        print(f"  {'[DRY RUN]' if dry_run else 'Uploading'} {local_path} → s3://{BUCKET}/{s3_key}")
        if not dry_run:
            s3_client.upload_file(local_path, BUCKET, s3_key)


def build_job_request(job_name, use_spot):
    # Resolve the correct DLC image URI via the SDK (avoids hardcoding tag formats)
    import sagemaker.image_uris
    image_uri = sagemaker.image_uris.retrieve(
        framework='pytorch',
        region=REGION,
        version=PYTORCH_VERSION,
        py_version=PYTHON_VERSION,
        instance_type=INSTANCE_TYPE,
        image_scope='training',
    )
    print(f"Image: {image_uri}")

    request = {
        'TrainingJobName': job_name,
        'RoleArn': ROLE_ARN,
        'AlgorithmSpecification': {
            'TrainingImage':     image_uri,
            'TrainingInputMode': 'File',
        },
        'HyperParameters': {
            **HYPERPARAMETERS,
            'sagemaker_program':       'sm_train_v3.py',
            'sagemaker_submit_directory': f's3://{BUCKET}/source/{job_name}/source.tar.gz',
        },
        'InputDataConfig': [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType':             'S3Prefix',
                        'S3Uri':                  f's3://{BUCKET}/{DATA_PREFIX}',
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                },
                'InputMode': 'File',
            },
            {
                'ChannelName': 'checkpoints',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType':             'S3Prefix',
                        'S3Uri':                  f's3://{BUCKET}/{INIT_PREFIX}',
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                },
                'InputMode': 'File',
            },
        ],
        'OutputDataConfig': {
            'S3OutputPath': f's3://{BUCKET}/{OUTPUT_PREFIX}',
        },
        'ResourceConfig': {
            'InstanceType':   INSTANCE_TYPE,
            'InstanceCount':  1,
            'VolumeSizeInGB': 200,
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': MAX_RUN_SECONDS,
        },
        'CheckpointConfig': {
            'S3Uri':      f's3://{BUCKET}/{CHECKPOINT_PREFIX}',
            'LocalPath':  '/opt/ml/checkpoints',
        },
    }

    if use_spot:
        request['EnableManagedSpotTraining'] = True
        request['StoppingCondition']['MaxWaitTimeInSeconds'] = MAX_WAIT_SECONDS

    return request


def main():
    args = parse_args()
    use_spot = not args.no_spot
    job_name = args.job_name or f'flow-stage2-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    print(f"\nJob name:   {job_name}")
    print(f"Instance:   {INSTANCE_TYPE}  ({'SPOT' if use_spot else 'ON-DEMAND'})")
    print(f"Data:       s3://{BUCKET}/{DATA_PREFIX}")
    print(f"Flow cache: s3://{BUCKET}/{FLOW_CACHE_PREFIX}")
    print(f"Output:     s3://{BUCKET}/{OUTPUT_PREFIX}")
    print(f"Role:       {ROLE_ARN}")

    if args.dry_run:
        print("\nDry run — not launching.")
        return

    boto_session = boto3.Session(region_name=REGION)
    s3_client   = boto_session.client('s3')
    sm_client   = boto_session.client('sagemaker')

    # Upload source code
    print("\nPackaging source...")
    import tarfile, tempfile
    source_files = [
        'sm_train_v3.py', 'full_scale_classifier.py', 'dataset.py',
        'sagemaker_requirements.txt',
    ]
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tmp_path = tmp.name
    with tarfile.open(tmp_path, 'w:gz') as tar:
        for f in source_files:
            if os.path.exists(f):
                tar.add(f)
            else:
                print(f"  WARNING: {f} not found — skipping")
    source_key = f"source/{job_name}/source.tar.gz"
    print(f"  Uploading source → s3://{BUCKET}/{source_key}")
    s3_client.upload_file(tmp_path, BUCKET, source_key)
    os.unlink(tmp_path)

    print("\nUploading init weights...")
    upload_init_weights(s3_client, force=args.force_init_upload)

    # Submit job
    request = build_job_request(job_name, use_spot)

    confirm = input(f"\nSubmit job '{job_name}'? [y/N] ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    print(f"\nSubmitting...")
    sm_client.create_training_job(**request)

    print(f"\nJob submitted: {job_name}")
    print(f"Monitor:  https://console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{job_name}")
    print(f"Logs:     https://console.aws.amazon.com/cloudwatch/home?region={REGION}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={job_name}")
    print(f"Status:   aws sagemaker describe-training-job --training-job-name {job_name} --region {REGION} --query TrainingJobStatus")


if __name__ == '__main__':
    main()
