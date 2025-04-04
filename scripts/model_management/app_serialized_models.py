#!/usr/bin/env python3
"""
Serialized Model Patch for app.py

This script adds a patch to app.py to load serialized models if available.
"""

import logging
import os
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def patch_app_for_serialized_models():
    """
    Patch app.py to use serialized models for faster loading
    
    This function adds code to app.py that checks for serialized models
    and loads them if available, instead of initializing them from scratch.
    """
    logger.info("Patching app.py to use serialized models...")
    
    app_path = Path('app.py')
    backup_path = Path('app.py.bak')
    
    # Create a backup of app.py
    if not backup_path.exists():
        with open(app_path, 'r') as f:
            content = f.read()
        
        with open(backup_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Created backup of app.py at {backup_path}")
    
    # Read the app.py file
    with open(app_path, 'r') as f:
        lines = f.readlines()
    
    # Find the place where models are initialized
    initialization_line = -1
    for i, line in enumerate(lines):
        if 'query_processor = MultiAspectQueryProcessor()' in line:
            initialization_line = i
            break
    
    if initialization_line == -1:
        logger.error("Could not find model initialization in app.py")
        return False
    
    # Add code to check for serialized models
    serialized_code = [
        "# Check for serialized models\n",
        "serialized_models_path = Path('serialized_models')\n",
        "if serialized_models_path.exists() and os.path.isfile('load_serialized_models.py'):\n",
        "    logger.info('Serialized models found, attempting to load...')\n",
        "    try:\n",
        "        import load_serialized_models\n",
        "        if load_serialized_models.check_serialized_models():\n",
        "            logger.info('Using serialized models for faster initialization')\n",
        "            # Load serialized models into global components\n",
        "            loaded_models = load_serialized_models.load_all_models()\n",
        "            # Memory manager and query processor will still be initialized normally\n",
        "        else:\n",
        "            logger.info('Serialized models check failed, using normal initialization')\n",
        "    except Exception as e:\n",
        "        logger.error(f'Error loading serialized models: {str(e)}')\n",
        "        logger.info('Falling back to normal initialization')\n",
        "else:\n",
        "    logger.info('No serialized models found, using normal initialization')\n",
        "\n"
    ]
    
    # Insert the code before model initialization
    updated_lines = lines[:initialization_line] + serialized_code + lines[initialization_line:]
    
    # Write the updated file
    with open(app_path, 'w') as f:
        f.writelines(updated_lines)
    
    logger.info("Successfully patched app.py to use serialized models")
    return True

def add_to_dockerfile():
    """
    Add serialized model loading to Dockerfile
    
    Ensures the Dockerfile copies serialized models and loads them
    during container initialization.
    """
    logger.info("Adding serialized model loading to Dockerfile...")
    
    dockerfile_path = Path('Dockerfile')
    if not dockerfile_path.exists():
        logger.warning("Dockerfile not found, creating a new one with serialized model support")
        
        # Create a new Dockerfile with serialized model support
        dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy serialized models if they exist
COPY serialized_models/ /app/serialized_models/

# Copy load script
COPY load_serialized_models.py .
COPY serialize_models.py .

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV USE_SERIALIZED_MODELS=true

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
"""
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        logger.info("Created new Dockerfile with serialized model support")
        return True
    
    # Dockerfile exists, modify it
    with open(dockerfile_path, 'r') as f:
        content = f.readlines()
    
    # Check if serialized models are already included
    if any('COPY serialized_models/' in line for line in content):
        logger.info("Dockerfile already includes serialized models")
        return True
    
    # Add serialized models to Dockerfile
    updated_content = []
    workdir_index = -1
    requirements_index = -1
    
    for i, line in enumerate(content):
        updated_content.append(line)
        
        if 'WORKDIR /app' in line:
            workdir_index = i
        
        if 'COPY requirements.txt' in line and i > workdir_index:
            requirements_index = i
    
    if requirements_index != -1:
        # Add after requirements are copied and installed
        install_index = -1
        for i in range(requirements_index, len(content)):
            if 'RUN pip install' in content[i]:
                install_index = i
                break
        
        if install_index != -1:
            # Add serialized models after requirements are installed
            serialized_lines = [
                "\n# Copy serialized models if they exist\n",
                "COPY serialized_models/ /app/serialized_models/\n",
                "\n# Copy model loading scripts\n",
                "COPY load_serialized_models.py .\n",
                "COPY serialize_models.py .\n",
                "\n# Set environment variable to use serialized models\n",
                "ENV USE_SERIALIZED_MODELS=true\n"
            ]
            
            updated_content = content[:install_index+1] + serialized_lines + content[install_index+1:]
            
            with open(dockerfile_path, 'w') as f:
                f.writelines(updated_content)
            
            logger.info("Updated Dockerfile to include serialized models")
            return True
    
    logger.warning("Could not find appropriate place to add serialized models in Dockerfile")
    return False

if __name__ == "__main__":
    # Patch app.py
    patched = patch_app_for_serialized_models()
    
    # Add to Dockerfile
    added_to_dockerfile = add_to_dockerfile()
    
    # Print results
    if patched and added_to_dockerfile:
        logger.info("Successfully prepared application for serialized models")
        sys.exit(0)
    else:
        logger.error("Failed to prepare application for serialized models")
        sys.exit(1) 