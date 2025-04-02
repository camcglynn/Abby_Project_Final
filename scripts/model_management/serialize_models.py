#!/usr/bin/env python3
"""
Model Serialization Script for Abby Chatbot

This script serializes the machine learning models used in the chatbot
for efficient deployment to AWS.
"""

import os
import torch
import pickle
import logging
import sys
import json
import numpy as np
from pathlib import Path
import nltk

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import from the project
sys.path.append('.')

# Create serialized_models directory if it doesn't exist
os.makedirs('serialized_models', exist_ok=True)

def download_nltk_data():
    """Download required NLTK data packages"""
    logger.info("Downloading NLTK data...")
    nltk_packages = [
        'punkt', 
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    for package in nltk_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
            logger.info(f"Package '{package}' already downloaded")
        except LookupError:
            logger.info(f"Downloading '{package}'...")
            nltk.download(package)

def serialize_bert_rag_model():
    """Serialize the BERT RAG model"""
    logger.info("Serializing BERT RAG model...")
    
    try:
        from chatbot.bert_rag import BertRAGModel
        
        # Initialize the model
        bert_rag = BertRAGModel()
        
        # Serialize model components
        output_path = Path('serialized_models/bert_rag')
        output_path.mkdir(exist_ok=True)
        
        # 1. Save the model configuration
        model_config = {
            'model_name': bert_rag.model_name,
            'qa_pairs_count': len(bert_rag.qa_pairs),
            'synonyms': bert_rag.synonyms
        }
        
        with open(output_path / 'model_config.json', 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # 2. Save the question embeddings using numpy to avoid pickle issues
        # Convert torch tensor to numpy array and save in numpy format
        if hasattr(bert_rag, 'question_embeddings') and isinstance(bert_rag.question_embeddings, torch.Tensor):
            embeddings_np = bert_rag.question_embeddings.cpu().detach().numpy()
            np.save(output_path / 'question_embeddings.npy', embeddings_np)
            logger.info(f"Saved embeddings with shape {embeddings_np.shape} to numpy format")
            
            # Also save in torch format as backup
            try:
                torch.save(bert_rag.question_embeddings, output_path / 'question_embeddings.pt')
            except Exception as e:
                logger.warning(f"Could not save torch embeddings, will rely on numpy format: {str(e)}")
        
        # 3. Save the FAISS index if available
        if hasattr(bert_rag, 'index'):
            faiss_path = output_path / 'faiss_index.bin'
            import faiss
            faiss.write_index(bert_rag.index, str(faiss_path))
        
        # 4. Save the QA pairs data
        with open(output_path / 'qa_pairs.json', 'w') as f:
            json.dump(bert_rag.qa_pairs, f, indent=2)
        
        logger.info(f"BERT RAG model serialized to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error serializing BERT RAG model: {str(e)}", exc_info=True)
        return False

def serialize_question_classifier():
    """Serialize the Question Classifier model"""
    logger.info("Serializing Question Classifier...")
    
    try:
        from chatbot.question_classifier import QuestionClassifier
        
        # Initialize the model
        classifier = QuestionClassifier()
        
        # Serialize model components
        output_path = Path('serialized_models/question_classifier')
        output_path.mkdir(exist_ok=True)
        
        # Save the model components - focus on any cached data or custom vocabulary
        if hasattr(classifier, 'category_keywords'):
            with open(output_path / 'category_keywords.json', 'w') as f:
                json.dump(classifier.category_keywords, f, indent=2)
        
        # Save any other serializable components
        model_data = {
            'state_patterns': classifier.state_patterns if hasattr(classifier, 'state_patterns') else {},
            'county_patterns': classifier.county_patterns if hasattr(classifier, 'county_patterns') else {},
            'zip_patterns': classifier.zip_patterns if hasattr(classifier, 'zip_patterns') else {}
        }
        
        with open(output_path / 'model_data.json', 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Question Classifier serialized to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error serializing Question Classifier: {str(e)}", exc_info=True)
        return False

def serialize_preprocessor():
    """Serialize the Preprocessor components"""
    logger.info("Serializing Preprocessor components...")
    
    try:
        from chatbot.preprocessor import Preprocessor
        
        # Initialize the preprocessor
        preprocessor = Preprocessor()
        
        # Serialize model components
        output_path = Path('serialized_models/preprocessor')
        output_path.mkdir(exist_ok=True)
        
        # Save regex patterns and other configuration
        components = {
            'email_pattern': preprocessor.email_pattern.pattern,
            'phone_pattern': preprocessor.phone_pattern.pattern,
            'zip_pattern': preprocessor.zip_pattern.pattern,
            'us_states': preprocessor.us_states,
            'has_zipcodes_library': hasattr(preprocessor, 'HAVE_ZIPCODES') and preprocessor.HAVE_ZIPCODES
        }
        
        with open(output_path / 'preprocessor_config.json', 'w') as f:
            json.dump(components, f, indent=2)
        
        # Save symspell dictionary if it exists
        if preprocessor.sym_spell and hasattr(preprocessor.sym_spell, 'words'):
            symspell_data = {
                'max_edit_distance': preprocessor.sym_spell._max_dictionary_edit_distance,
                'prefix_length': preprocessor.sym_spell._prefix_length,
                'words': {word: freq for word, freq in preprocessor.sym_spell.words.items()}
            }
            
            with open(output_path / 'symspell_data.json', 'w') as f:
                json.dump(symspell_data, f, indent=2)
        
        logger.info(f"Preprocessor components serialized to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error serializing Preprocessor: {str(e)}", exc_info=True)
        return False

def main():
    """Main execution function"""
    logger.info("Starting model serialization process")
    
    # Download required NLTK data
    download_nltk_data()
    
    # Serialize models
    results = {
        "bert_rag": serialize_bert_rag_model(),
        "question_classifier": serialize_question_classifier(),
        "preprocessor": serialize_preprocessor()
    }
    
    # Report results
    logger.info("Model serialization complete")
    for model, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{model}: {status}")
    
    # Output serialization info to a status file
    with open('serialized_models/serialization_info.json', 'w') as f:
        json.dump({
            "timestamp": str(logging.Formatter().converter()),
            "results": results,
            "serialized_models_path": os.path.abspath('serialized_models')
        }, f, indent=2)
    
    logger.info("Serialization info saved to serialized_models/serialization_info.json")

if __name__ == "__main__":
    main() 