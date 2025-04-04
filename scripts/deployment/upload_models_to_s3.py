#!/usr/bin/env python3
"""
Upload ML models and NLTK data to S3 for serverless deployment
This ensures Lambda functions can download models at runtime instead of packaging them
"""

import os
import sys
import logging
import argparse
import boto3
import nltk
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Upload ML models to S3')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--stage', type=str, default='dev', help='Deployment stage (dev, staging, prod)')
    parser.add_argument('--download-nltk', action='store_true', help='Download NLTK data')
    return parser.parse_args()

def download_nltk_data():
    """Download required NLTK datasets"""
    logger.info("Downloading NLTK data...")
    nltk_data_dir = 'nltk_data'
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Download required NLTK datasets
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)
    
    logger.info("NLTK data downloaded successfully")
    return nltk_data_dir

def upload_directory_to_s3(s3_client, local_dir, bucket, prefix):
    """Upload a directory to S3 recursively"""
    if not os.path.exists(local_dir):
        logger.warning(f"Directory {local_dir} does not exist. Skipping.")
        return

    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, start=os.path.dirname(local_dir))
            s3_path = os.path.join(prefix, relative_path).replace('\\', '/')
            
            logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_path}")
            s3_client.upload_file(local_path, bucket, s3_path)

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Validate bucket name
    if not args.bucket:
        logger.error("Bucket name is required")
        sys.exit(1)
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3')
        # Check if bucket exists
        s3_client.head_bucket(Bucket=args.bucket)
    except Exception as e:
        logger.error(f"Error connecting to S3 bucket {args.bucket}: {str(e)}")
        sys.exit(1)
    
    logger.info(f"Connected to S3 bucket: {args.bucket}")
    
    # Download NLTK data if requested
    if args.download_nltk:
        nltk_data_dir = download_nltk_data()
        # Upload NLTK data to S3
        upload_directory_to_s3(
            s3_client, 
            nltk_data_dir, 
            args.bucket, 
            f"nltk_data/{args.stage}"
        )
    
    # Upload models directory if it exists
    models_dir = 'models'
    if os.path.exists(models_dir):
        logger.info(f"Uploading models directory to S3...")
        upload_directory_to_s3(
            s3_client, 
            models_dir, 
            args.bucket, 
            f"models/{args.stage}"
        )
    else:
        logger.warning(f"Models directory '{models_dir}' not found")
    
    logger.info("Upload process completed successfully")

if __name__ == "__main__":
    main() 