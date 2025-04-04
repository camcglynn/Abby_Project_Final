#!/bin/bash

# Exit on error
set -e

# Logging configuration
exec > >(tee /var/log/codedeploy-before-install.log) 2>&1
echo "Starting BeforeInstall script at $(date)"

# Update system packages
yum update -y

# Install Python 3.9 and development tools
amazon-linux-extras enable python3.9
yum install -y python39 python39-devel gcc

# Create application directory if it doesn't exist
mkdir -p /home/ec2-user/abby-chatbot

# Install pip for Python 3.9
curl -O https://bootstrap.pypa.io/get-pip.py
python3.9 get-pip.py
rm get-pip.py

# Create virtual environment
python3.9 -m venv /home/ec2-user/abby-chatbot/venv

# Install system dependencies
yum install -y git

# Create application directory if it doesn't exist
APP_DIR="/opt/abby-chatbot"
DATA_DIR="$APP_DIR/data"
MODELS_DIR="$APP_DIR/serialized_models"
mkdir -p $APP_DIR $DATA_DIR $MODELS_DIR

# Clean up old files if they exist
if [ -d "$APP_DIR" ]; then
    # Preserve data and model directories
    mv $DATA_DIR /tmp/data_backup
    mv $MODELS_DIR /tmp/models_backup
    
    # Clean up application files
    rm -rf $APP_DIR/*
    
    # Restore data and model directories
    mv /tmp/data_backup $DATA_DIR
    mv /tmp/models_backup $MODELS_DIR
fi

# Create virtual environment directory if it doesn't exist
VENV_DIR="$APP_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

# Download NLTK data
NLTK_DIR="$APP_DIR/nltk_data"
mkdir -p $NLTK_DIR
export NLTK_DATA=$NLTK_DIR

# Set correct ownership
chown -R ec2-user:ec2-user $APP_DIR

echo "BeforeInstall script completed successfully"
exit 0