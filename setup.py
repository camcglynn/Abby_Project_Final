#!/usr/bin/env python3
"""
Setup script for Abby Chatbot.
This script initializes the project structure.
"""

import os
import logging
import argparse
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("setup")

def download_nltk_data():
    """Download required NLTK data."""
    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    
    nltk.data.path.append(nltk_data_path)
    
    required_packages = ['punkt', 'stopwords', 'wordnet']
    for package in required_packages:
        try:
            nltk.download(package, download_dir=nltk_data_path)
            logger.info(f"Downloaded NLTK package: {package}")
        except Exception as e:
            logger.error(f"Failed to download NLTK package {package}: {str(e)}")

def create_directory_structure():
    """Create the necessary directory structure for the project."""
    directories = [
        'data',
        'logs',
        'instance',
        'static/css',
        'static/js',
        'static/images',
        'templates',
        'templates/admin',
        'chatbot',
        'models',
        'utils',
        'uploads'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")

def create_empty_data_files():
    """Create empty data directory files to establish structure."""
    data_files = [
        'data/policy_data.json',
        'data/clinics_data.json'
    ]
    
    for file_path in data_files:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create empty data file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write('{}')
            logger.info(f"Created empty data file: {file_path}")
        else:
            logger.info(f"Data file already exists: {file_path}")

def create_init_files():
    """Create __init__.py files in necessary directories."""
    init_dirs = ['chatbot', 'models', 'utils']
    
    for directory in init_dirs:
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# {directory} package initialization\n")
            logger.info(f"Created init file: {init_file}")
        else:
            logger.info(f"Init file already exists: {init_file}")

def create_empty_database():
    """Create an empty SQLite database file."""
    db_file = 'instance/abby_chatbot.db'
    
    if os.path.exists(db_file):
        logger.info(f"Database file already exists: {db_file}")
        return
    
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    
    # Create an empty file
    with open(db_file, 'w') as f:
        pass
    
    logger.info(f"Created empty database file: {db_file}")

def main():
    """Execute the setup process."""
    parser = argparse.ArgumentParser(description='Set up the Abby Chatbot project')
    parser.add_argument('--skip-empty-files', action='store_true', help='Skip creating empty data files')
    args = parser.parse_args()
    
    logger.info("Starting Abby Chatbot setup...")
    
    # Create directory structure
    create_directory_structure()
    
    # Create __init__.py files
    create_init_files()
    
    # Create empty database
    create_empty_database()
    
    # Create empty data files if not skipped
    if not args.skip_empty_files:
        create_empty_data_files()
    
    # Download NLTK data
    download_nltk_data()
    
    logger.info("Setup completed successfully.")
    print("\nSetup completed successfully!")
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up environment variables (see README.md)")
    print("3. Add your own data to the data files")
    print("4. Run the application: flask run")

if __name__ == "__main__":
    main() 