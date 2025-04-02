#!/bin/bash

# Exit on error
set -e

# Logging configuration
exec > >(tee /var/log/codedeploy-after-install.log) 2>&1
echo "Starting AfterInstall script at $(date)"

# Set environment variables
APP_DIR="/home/ec2-user/abby-chatbot"
VENV_DIR="$APP_DIR/venv"
DATA_DIR="$APP_DIR/data"
MODELS_DIR="$APP_DIR/serialized_models"
NLTK_DIR="$APP_DIR/nltk_data"
export NLTK_DATA=$NLTK_DIR

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Install requirements
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r $APP_DIR/requirements.txt

# Install additional packages needed for deployment
pip install gunicorn

# Download required NLTK data
echo "Setting up NLTK data..."
python3 -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DIR'); nltk.download('averaged_perceptron_tagger', download_dir='$NLTK_DIR'); nltk.download('wordnet', download_dir='$NLTK_DIR')"

# Run data setup script
chmod +x $APP_DIR/scripts/setup_data.sh
$APP_DIR/scripts/setup_data.sh

# Set correct permissions
echo "Setting file permissions..."
sudo chown -R ec2-user:ec2-user $APP_DIR
sudo chmod -R 755 $APP_DIR

echo "Creating log directories..."
sudo mkdir -p /var/log/abby-chatbot
sudo chown -R ec2-user:ec2-user /var/log/abby-chatbot

echo "Setting up model data..."
if [ ! -d "$APP_DIR/serialized_models" ]; then
    echo "Downloading model data from S3..."
    aws s3 cp s3://abby-chatbot-models/serialized_models.tar.gz /tmp/
    tar -xzf /tmp/serialized_models.tar.gz -C $APP_DIR
    rm /tmp/serialized_models.tar.gz
fi

echo "Setting up environment variables..."
if [ ! -f $APP_DIR/.env ]; then
    echo "Creating .env file from SSM parameters..."
    # Get parameters from AWS Systems Manager Parameter Store
    aws ssm get-parameter --name "/abby-chatbot/prod/OPENAI_API_KEY" --with-decryption --query "Parameter.Value" --output text > $APP_DIR/.env
    aws ssm get-parameter --name "/abby-chatbot/prod/ABORTION_POLICY_API_KEY" --with-decryption --query "Parameter.Value" --output text >> $APP_DIR/.env
    # Add other environment variables as needed
fi

echo "AfterInstall script completed successfully"
exit 0