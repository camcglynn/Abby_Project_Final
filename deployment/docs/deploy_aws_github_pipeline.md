# Deploying Abby Chatbot Using AWS + GitHub CI/CD Pipeline

This guide explains how to deploy your chatbot using the AWS infrastructure and GitHub CI/CD pipeline as shown in the diagram.

## Prerequisites

1. AWS Account with appropriate permissions
2. GitHub repository with your chatbot code
3. AWS CLI installed and configured
4. GitHub personal access token

## Step 1: Prepare Your Repository

Ensure your chatbot code is in a GitHub repository with the following structure:
- `app.py` - Main application file
- `requirements.txt` - Dependencies
- `chatbot/` - Chatbot modules
- `.env.example` - Example environment variables (do not commit actual .env file)
- `serverless.yml` - For AWS serverless deployment
- `lambda_handler.py` - AWS Lambda handler

## Step 2: Set Up AWS Infrastructure

### 1. Create VPC and Network Components

Create a VPC with the components shown in the diagram:
- Public/Private Subnets
- Route Tables
- Internet Gateway
- NAT Gateway

```bash
# Use AWS CloudFormation or AWS CDK to create the VPC infrastructure
aws cloudformation create-stack --stack-name abby-chatbot-vpc --template-body file://vpc-template.yml
```

### 2. Set Up Security Groups

Create security groups for:
- ALBSecurityGroup
- WebappSecurityGroup

```bash
# Create ALB Security Group
aws ec2 create-security-group --group-name ALBSecurityGroup \
  --description "Security group for the Application Load Balancer" \
  --vpc-id <vpc-id>

# Create Webapp Security Group
aws ec2 create-security-group --group-name WebappSecurityGroup \
  --description "Security group for the web application" \
  --vpc-id <vpc-id>
```

### 3. Create S3 Bucket for Deployment

```bash
# Create S3 bucket for deployment artifacts
aws s3 create-bucket --bucket abby-chatbot-deployment-bucket
```

### 4. Create IAM Roles

Create the GitHubIAMRole and IDCProvider roles:

```bash
# Create IAM role for GitHub Actions
aws iam create-role --role-name GitHubIAMRole \
  --assume-role-policy-document file://github-role-policy.json

# Attach necessary policies
aws iam attach-role-policy --role-name GitHubIAMRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-role-policy --role-name GitHubIAMRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonECS-FullAccess
```

## Step 3: Set Up GitHub Actions Workflow

Create a `.github/workflows/deploy.yml` file in your repository:

```yaml
name: Deploy Abby Chatbot

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build, tag, and push image to Amazon ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: abby-chatbot
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
      
      - name: Deploy to AWS
        run: |
          aws s3 cp deployment-package.zip s3://abby-chatbot-deployment-bucket/
          aws cloudformation deploy \
            --template-file cloudformation/webapp.yml \
            --stack-name abby-chatbot-webapp \
            --parameter-overrides \
              ImageTag=${{ github.sha }} \
              Environment=production
```

## Step 4: Create Dockerfile and Deployment Configuration

### Dockerfile

Create a `Dockerfile` in your repository:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### AWS CloudFormation Template

Create a `cloudformation/webapp.yml` template that defines:
- WebappApplication
- WebappDeploymentGroup
- WebappLaunchConfig
- ApplicationLoadBalancer

## Step 5: Configure Environment Variables

Store your environment variables in AWS Systems Manager Parameter Store:

```bash
# Store environment variables
aws ssm put-parameter --name "/abby-chatbot/prod/OPENAI_API_KEY" --value "your-api-key" --type SecureString
aws ssm put-parameter --name "/abby-chatbot/prod/ABORTION_POLICY_API_KEY" --value "your-api-key" --type SecureString
# Add other environment variables from your .env file
```

## Step 6: Deploy the Application

1. Push your code to GitHub main branch
2. The GitHub Actions workflow will build and deploy your application
3. Monitor the deployment in the GitHub Actions tab and AWS Console

## Step 7: Access Your Chatbot

After deployment completes, you can access your chatbot at the Application Load Balancer's DNS name:

```bash
# Get the ALB DNS name
aws elbv2 describe-load-balancers --query "LoadBalancers[*].DNSName" --output text
```

## Additional Configuration

### Scaling Configuration

To configure auto-scaling for your application:

```bash
# Create an Auto Scaling Group
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name abby-chatbot-asg \
  --launch-configuration-name abby-chatbot-launch-config \
  --min-size 2 \
  --max-size 5 \
  --desired-capacity 2 \
  --vpc-zone-identifier "subnet-id-1,subnet-id-2" \
  --target-group-arns "arn:aws:elasticloadbalancing:region:account-id:targetgroup/abby-chatbot-targets/abcdef123456"
```

### Setting Up DynamoDB Tables

If you're using DynamoDB (as in your serverless.yml):

```bash
# Create DynamoDB tables
aws dynamodb create-table \
  --table-name abby-chatbot-conversations-prod \
  --attribute-definitions AttributeName=conversation_id,AttributeType=S \
  --key-schema AttributeName=conversation_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

aws dynamodb create-table \
  --table-name abby-chatbot-users-prod \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

aws dynamodb create-table \
  --table-name abby-chatbot-feedback-prod \
  --attribute-definitions AttributeName=feedback_id,AttributeType=S \
  --key-schema AttributeName=feedback_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

## Troubleshooting

1. Check CloudWatch Logs for application errors
2. Verify security group settings to ensure traffic can flow to your application
3. Check IAM permissions if deployment fails
4. Verify environment variables are correctly set
5. Check that your S3 bucket for deployment artifacts is correctly configured 