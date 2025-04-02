#!/usr/bin/env python3
"""
Script to generate a buildspec.yml for AWS CodeBuild that creates a new Docker image 
with the Path import fixed, without using local disk space.
"""

import os
import yaml

# Define the buildspec content
buildspec = {
    'version': '0.2',
    'phases': {
        'pre_build': {
            'commands': [
                'echo Logging in to Amazon ECR...',
                'aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com',
                'REPOSITORY_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/abby-chatbot-v7',
                'echo Creating Dockerfile...',
                'echo "FROM $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/abby-chatbot-v6:latest" > Dockerfile',
                'echo "WORKDIR /app" >> Dockerfile',
                'echo "RUN grep -q \"from pathlib import Path\" app.py || sed -i \'1s/^/from pathlib import Path\\n/\' app.py" >> Dockerfile',
                'echo "CMD [\\"gunicorn\\", \\"--bind\\", \\"0.0.0.0:5000\\", \\"--workers\\", \\"4\\", \\"--timeout\\", \\"120\\", \\"app:app\\"]" >> Dockerfile',
                'cat Dockerfile'
            ]
        },
        'build': {
            'commands': [
                'echo Build started on `date`',
                'echo Building the Docker image...',
                'docker build -t $REPOSITORY_URI:latest .',
                'docker tag $REPOSITORY_URI:latest $REPOSITORY_URI:$CODEBUILD_RESOLVED_SOURCE_VERSION'
            ]
        },
        'post_build': {
            'commands': [
                'echo Build completed on `date`',
                'echo Pushing the Docker image...',
                'docker push $REPOSITORY_URI:latest',
                'docker push $REPOSITORY_URI:$CODEBUILD_RESOLVED_SOURCE_VERSION',
                'echo Creating imagedefinitions.json...',
                'echo \'[{"name":"abby-chatbot-v7-container-prod","imageUri":"\'$REPOSITORY_URI:latest\'"}]\' > imagedefinitions.json',
                'cat imagedefinitions.json'
            ]
        }
    },
    'artifacts': {
        'files': ['imagedefinitions.json']
    }
}

# Write to buildspec.yml
with open('buildspec.yml', 'w') as f:
    yaml.dump(buildspec, f, default_flow_style=False)

print("Generated buildspec.yml for AWS CodeBuild")
print("To use this, you would:")
print("1. Create an AWS CodeBuild project")
print("2. Upload this buildspec.yml to a git repository or S3 bucket")
print("3. Configure the project to use this buildspec")
print("4. Set environment variables:")
print("   - AWS_ACCOUNT_ID: Your AWS account ID")
print("   - AWS_DEFAULT_REGION: Your AWS region")
print("5. Run the build to create a fixed Docker image") 