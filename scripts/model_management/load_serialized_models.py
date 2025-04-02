#!/usr/bin/env python3
"""
Load Serialized Models Script for Abby Chatbot

This script loads the serialized machine learning models during deployment.
Use in AWS deployment to quickly initialize the chatbot with pre-trained models.
"""

import os
import torch
import logging
import sys
import json
import re
import numpy as np
from pathlib import Path
import importlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import from the project
sys.path.append('.')

def load_bert_rag_model(model_instance=None):
    """
    Load the serialized BERT RAG model
    
    Args:
        model_instance: Optional existing model instance to load into
        
    Returns:
        Loaded BertRAGModel instance or None if loading failed
    """
    logger.info("Loading serialized BERT RAG model...")
    
    try:
        from chatbot.bert_rag import BertRAGModel
        
        # Create model instance if not provided
        if model_instance is None:
            model_instance = BertRAGModel()
            # Skip the normal initialization process by setting a flag
            model_instance._skip_init = True
        
        # Load from serialized files
        model_path = Path('serialized_models/bert_rag')
        
        # 1. Load the model configuration
        with open(model_path / 'model_config.json', 'r') as f:
            model_config = json.load(f)
            
        # 2. Load the question embeddings - try numpy format first
        embeddings_loaded = False
        
        # Try loading from numpy format (preferred)
        numpy_path = model_path / 'question_embeddings.npy'
        if numpy_path.exists():
            try:
                embeddings_np = np.load(numpy_path)
                model_instance.question_embeddings = torch.from_numpy(embeddings_np)
                logger.info(f"Loaded embeddings from numpy format with shape {embeddings_np.shape}")
                embeddings_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load embeddings from numpy format: {str(e)}")
        
        # If numpy format failed, try torch format
        if not embeddings_loaded:
            torch_path = model_path / 'question_embeddings.pt'
            if torch_path.exists():
                try:
                    # First attempt with safety enabled (weights_only=True)
                    model_instance.question_embeddings = torch.load(torch_path)
                    embeddings_loaded = True
                except Exception as e:
                    logger.warning(f"Standard torch.load failed: {str(e)}")
                    # Try with safety bypassed for trusted data (our own serialized data)
                    logger.info("Attempting to load with weights_only=False (trusted data)")
                    try:
                        torch.serialization.add_safe_globals(['numpy.core.multiarray._reconstruct'])
                        model_instance.question_embeddings = torch.load(torch_path, weights_only=False)
                        embeddings_loaded = True
                    except Exception as e2:
                        logger.warning(f"Alternative torch.load also failed: {str(e2)}")
        
        # If all loading methods failed, regenerate embeddings
        if not embeddings_loaded:
            logger.info("Failed to load embeddings. Regenerating from QA pairs data")
            with open(model_path / 'qa_pairs.json', 'r') as f:
                qa_pairs = json.load(f)
                questions = [qa['Question'] for qa in qa_pairs]
                model_instance.question_embeddings = model_instance.generate_embeddings(questions)
        
        # 3. Load the FAISS index if available
        faiss_path = model_path / 'faiss_index.bin'
        if faiss_path.exists():
            import faiss
            model_instance.index = faiss.read_index(str(faiss_path))
        
        # 4. Load the QA pairs data
        with open(model_path / 'qa_pairs.json', 'r') as f:
            model_instance.qa_pairs = json.load(f)
        
        # 5. Set config values
        model_instance.model_name = model_config['model_name']
        model_instance.synonyms = model_config['synonyms']
        
        logger.info(f"BERT RAG model loaded successfully with {len(model_instance.qa_pairs)} QA pairs")
        return model_instance
    except Exception as e:
        logger.error(f"Error loading BERT RAG model: {str(e)}", exc_info=True)
        return None

def load_question_classifier(classifier_instance=None):
    """
    Load the serialized Question Classifier
    
    Args:
        classifier_instance: Optional existing classifier instance to load into
        
    Returns:
        Loaded QuestionClassifier instance or None if loading failed
    """
    logger.info("Loading serialized Question Classifier...")
    
    try:
        from chatbot.question_classifier import QuestionClassifier
        
        # Create classifier instance if not provided
        if classifier_instance is None:
            classifier_instance = QuestionClassifier()
            # Skip the normal initialization process if needed
            classifier_instance._skip_init = True
        
        # Load from serialized files
        model_path = Path('serialized_models/question_classifier')
        
        # Load category keywords if available
        keywords_path = model_path / 'category_keywords.json'
        if keywords_path.exists():
            with open(keywords_path, 'r') as f:
                classifier_instance.category_keywords = json.load(f)
        
        # Load other model data
        with open(model_path / 'model_data.json', 'r') as f:
            model_data = json.load(f)
            
            # Set attributes from model data
            classifier_instance.state_patterns = model_data.get('state_patterns', {})
            classifier_instance.county_patterns = model_data.get('county_patterns', {})
            classifier_instance.zip_patterns = model_data.get('zip_patterns', {})
        
        logger.info("Question Classifier loaded successfully")
        return classifier_instance
    except Exception as e:
        logger.error(f"Error loading Question Classifier: {str(e)}", exc_info=True)
        return None

def load_preprocessor(preprocessor_instance=None):
    """
    Load the serialized Preprocessor
    
    Args:
        preprocessor_instance: Optional existing preprocessor instance to load into
        
    Returns:
        Loaded Preprocessor instance or None if loading failed
    """
    logger.info("Loading serialized Preprocessor...")
    
    try:
        from chatbot.preprocessor import Preprocessor
        
        # Create preprocessor instance if not provided
        if preprocessor_instance is None:
            preprocessor_instance = Preprocessor()
        
        # Load from serialized files
        config_path = Path('serialized_models/preprocessor/preprocessor_config.json')
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                # Set configuration values
                preprocessor_instance.email_pattern = re.compile(config.get('email_pattern', r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'))
                preprocessor_instance.phone_pattern = re.compile(config.get('phone_pattern', r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'))
                preprocessor_instance.zip_pattern = re.compile(config.get('zip_pattern', r'\b(\d{5})\b'))
                
                # Update US states list if available
                if 'us_states' in config:
                    preprocessor_instance.us_states = config['us_states']
        
        # Load symspell data if available
        symspell_path = Path('serialized_models/preprocessor/symspell_data.json')
        if symspell_path.exists() and preprocessor_instance.sym_spell:
            with open(symspell_path, 'r') as f:
                symspell_data = json.load(f)
                
                # Update symspell configuration
                if hasattr(preprocessor_instance.sym_spell, 'words'):
                    # Load dictionary words
                    for word, freq in symspell_data.get('words', {}).items():
                        preprocessor_instance.sym_spell.words[word] = freq
        
        logger.info("Preprocessor loaded successfully")
        return preprocessor_instance
    except Exception as e:
        logger.error(f"Error loading Preprocessor: {str(e)}", exc_info=True)
        return None

def load_all_models():
    """
    Load all serialized models
    
    Returns:
        dict: Dictionary containing all loaded model instances
    """
    logger.info("Loading all serialized models...")
    
    models = {
        "bert_rag": load_bert_rag_model(),
        "question_classifier": load_question_classifier(),
        "preprocessor": load_preprocessor()
    }
    
    # Check which models were loaded successfully
    loaded_models = {name: model for name, model in models.items() if model is not None}
    failed_models = [name for name, model in models.items() if model is None]
    
    logger.info(f"Successfully loaded {len(loaded_models)} models: {list(loaded_models.keys())}")
    if failed_models:
        logger.warning(f"Failed to load {len(failed_models)} models: {failed_models}")
    
    return models

def check_serialized_models():
    """
    Check if the serialized models exist and are valid
    
    Returns:
        bool: True if all required serialized models exist
    """
    logger.info("Checking for serialized models...")
    
    # Define the required model directories
    required_dirs = [
        'serialized_models/bert_rag',
        'serialized_models/question_classifier',
        'serialized_models/preprocessor'
    ]
    
    # Check if directories exist
    missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]
    
    if missing_dirs:
        logger.warning(f"Missing serialized model directories: {missing_dirs}")
        return False
    
    # Check for info file
    info_file = 'serialized_models/serialization_info.json'
    if not os.path.isfile(info_file):
        logger.warning(f"Missing serialization info file: {info_file}")
        return False
    
    logger.info("All required serialized model directories exist")
    return True

if __name__ == "__main__":
    # Check if serialized models exist
    if check_serialized_models():
        # Load all models
        models = load_all_models()
        
        # Print status
        for name, model in models.items():
            status = "LOADED" if model is not None else "FAILED"
            logger.info(f"{name}: {status}")
        
        # Exit with error code if any model failed to load
        if any(model is None for model in models.values()):
            sys.exit(1)
    else:
        logger.error("Serialized models not found or incomplete")
        sys.exit(1) 