#!/usr/bin/env bash
set -euo pipefail

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "uv installed successfully."

# Prompt for email for SSH key generation
read -r -p "Enter your email for SSH key generation: " user_email
if [[ -z "$user_email" ]]; then
    echo "Error: Email cannot be empty."
    exit 1
fi

echo "Generating SSH key..."
ssh-keygen -t rsa -b 4096 -C "$user_email"

echo "SSH key generated successfully."

cd /root/.ssh
eval $(ssh-agent -s)
ssh-add id_rsa
cat id_rsa.pub