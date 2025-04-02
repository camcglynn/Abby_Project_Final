#!/usr/bin/env python
"""
Demo script for the Preprocessor integration with the chatbot pipeline.
"""

import logging
import sys
import os
import json
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("preprocessor_demo")

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required classes
try:
    from chatbot import MultiAspectQueryProcessor
    logger.info("Successfully imported MultiAspectQueryProcessor")
except ImportError as e:
    logger.error(f"Failed to import MultiAspectQueryProcessor: {e}")
    sys.exit(1)

async def process_message(processor, message):
    """Process a message through the pipeline"""
    logger.info(f"Processing message: {message}")
    
    # Create empty conversation history for the demo
    conversation_history = []
    
    # Process the message
    result = await processor.process_query(message, conversation_history)
    
    # Print the result
    logger.info(f"Processed message: {result['processed_message'] if 'processed_message' in result else message}")
    logger.info(f"Preprocessing metadata: {json.dumps(result.get('preprocessing_metadata', {}), indent=2)}")
    logger.info(f"Response: {result['text']}")
    
    if "state_code" in result:
        logger.info(f"Detected state: {result['state_code']}")
    
    return result

async def run_demo():
    """Run the preprocessor demo"""
    logger.info("Initializing MultiAspectQueryProcessor...")
    processor = MultiAspectQueryProcessor()
    
    # Test messages
    messages = [
        # Language detection (non-English)
        "¿Cuáles son las leyes de aborto en Texas?",
        
        # PII redaction
        "My name is Jane and my email is jane@example.com. What are the abortion laws in Texas?",
        
        # PII redaction with phone
        "Please call me at (555) 123-4567 to discuss abortion options.",
        
        # Typo correction (short message)
        "Where can I get an aborton?",
        
        # ZIP code detection
        "I live in 90210, what are my options?",
        
        # Complete test with multiple features
        "My phone is (555) 987-6543 and I live in 77429. I need to know about aborton laws."
    ]
    
    # Process each message
    for i, message in enumerate(messages):
        logger.info(f"\n===== Test {i+1}: {message} =====")
        await process_message(processor, message)

def main():
    """Main entry point"""
    logger.info("Starting Preprocessor demo...")
    
    try:
        asyncio.run(run_demo())
        logger.info("\nDemo completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error during demo: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 