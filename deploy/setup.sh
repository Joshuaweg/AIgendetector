#!/usr/bin/env bash
# setup.sh — one-shot bootstrap for a fresh AWS Deep Learning AMI (Ubuntu)
# Run as ubuntu: bash setup.sh
set -euo pipefail

REPO_DIR="/home/ubuntu/AIgendetector"
SERVICE_NAME="api_server"
LOG_DIR="/var/log/api_server"

echo "=== 1. System packages ==="
sudo apt-get update -q
sudo apt-get install -y nginx ffmpeg python3-venv python3-pip

echo "=== 2. Log directory ==="
sudo mkdir -p "$LOG_DIR"
sudo chown ubuntu:ubuntu "$LOG_DIR"

echo "=== 3. Python virtual environment ==="
cd "$REPO_DIR"
python3 -m venv venv
venv/bin/pip install --upgrade pip --quiet
venv/bin/pip install -r requirements.txt --quiet
echo "Dependencies installed."

echo "=== 4. Persistent data directories ==="
mkdir -p saved_videos saved_attributions feedback temp_uploads

echo "=== 5. Nginx ==="
sudo cp deploy/nginx.conf /etc/nginx/sites-available/"$SERVICE_NAME"
sudo ln -sf /etc/nginx/sites-available/"$SERVICE_NAME" \
            /etc/nginx/sites-enabled/"$SERVICE_NAME"
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx
echo "Nginx configured."

echo "=== 6. Systemd service ==="
sudo cp deploy/api_server.service /etc/systemd/system/"$SERVICE_NAME".service
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl start  "$SERVICE_NAME"
echo "Service started. Check status:"
echo "  sudo systemctl status $SERVICE_NAME"
echo "  sudo journalctl -u $SERVICE_NAME -f"

echo ""
echo "=== 7. IAM role reminder ==="
echo "Attach an IAM role to this instance with the following inline policy"
echo "so the idle watchdog can stop itself:"
echo ""
cat <<'IAM'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ec2:StopInstances",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "ec2:ResourceTag/Name": "ai-video-detector"
        }
      }
    }
  ]
}
IAM

echo ""
echo "=== Done ==="
echo "API running at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "Set NEXT_PUBLIC_CLASSIFIER_API_URL to that address in Amplify env vars."
