# Abby Chatbot

A specialized chatbot designed to provide accurate and supportive information about reproductive healthcare.

## Project Structure

```
.
├── app.py                 # Main FastAPI application
├── chatbot/              # Core chatbot implementation
├── config/               # Configuration files
├── data/                # Data files (managed separately)
├── deployment/          # Deployment configuration
│   ├── cloudformation/  # AWS CloudFormation templates
│   ├── docs/           # Deployment documentation
│   ├── scripts/        # Deployment scripts
│   └── terraform/      # Terraform configuration
├── models/             # ML model implementations
├── nltk_data/         # NLTK data files (managed separately)
├── scripts/           # Utility scripts
├── serialized_models/ # Pre-trained models (managed separately)
├── static/           # Static web assets
├── templates/        # HTML templates
├── tests/           # Test suite
└── utils/           # Utility functions
```

## Prerequisites

- Python 3.8+
- pip
- virtualenv or conda
- AWS CLI (for deployment)

## Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sahanasankar13/Abby_Chatbot.git
   cd Abby_Chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run the application:
   ```bash
   python app.py
   ```

The application will be available at http://localhost:5006

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Deployment

### AWS Deployment

1. Configure AWS credentials:
   ```bash
   aws configure
   ```

2. Deploy using CloudFormation:
   ```bash
   cd deployment/cloudformation
   aws cloudformation create-stack --stack-name abby-chatbot --template-body file://abby-chatbot-stack.yml --capabilities CAPABILITY_NAMED_IAM
   ```

For detailed deployment instructions, see [deployment/docs/deploy_aws_github_pipeline.md](deployment/docs/deploy_aws_github_pipeline.md)

## Data and Model Management

The following directories are managed separately and not included in version control:
- `data/`: Contains training data and configuration files
- `serialized_models/`: Contains pre-trained models
- `nltk_data/`: Contains NLTK data files

Contact the project maintainers for access to these resources.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is proprietary and confidential. All rights reserved.

## Contact

For questions or support, please contact the project maintainers.