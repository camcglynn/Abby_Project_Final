#!/bin/bash

# Exit on error
set -e

# Logging configuration
exec > >(tee /var/log/codedeploy-start-application.log) 2>&1
echo "Starting application script at $(date)"

# Set environment variables
APP_DIR="/opt/abby-chatbot"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="/var/log/abby-chatbot"

# Create log directory if it doesn't exist
sudo mkdir -p /var/log/abby-chatbot
sudo chown -R ec2-user:ec2-user /var/log/abby-chatbot

# Copy systemd service file
sudo cp /home/ec2-user/abby-chatbot/deployment/scripts/abby-chatbot.service /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Start the service
sudo systemctl start abby-chatbot

# Enable the service to start on boot
sudo systemctl enable abby-chatbot

# Wait for the service to start
sleep 5

# Check service status
sudo systemctl status abby-chatbot

echo "Application start script completed successfully"
exit 0 