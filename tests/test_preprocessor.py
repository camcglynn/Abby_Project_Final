#!/usr/bin/env python
"""
Test script for the Preprocessor module.
This validates all the preprocessing features:
1. Language detection
2. PII redaction
3. Typo correction
4. ZIP code lookup
"""

import logging
import sys
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("preprocessor_test")

# Add the current directory to the path so we can import the chatbot module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from chatbot.preprocessor import Preprocessor
    logger.info("Successfully imported Preprocessor")
except ImportError as e:
    logger.error(f"Failed to import Preprocessor: {e}")
    sys.exit(1)

def test_language_detection():
    """Test the language detection functionality"""
    logger.info("\n=== Testing Language Detection ===")
    
    preprocessor = Preprocessor()
    
    # Test English
    test_message = "How do I find an abortion clinic near me?"
    result = preprocessor.process(test_message)
    logger.info(f"English test: {result['is_processable']} - {result['metadata']['preprocessing']['language_check']}")
    
    # Test Spanish
    test_message = "¿Dónde puedo encontrar una clínica de aborto cerca de mí?"
    result = preprocessor.process(test_message)
    logger.info(f"Spanish test: {result['is_processable']} - {result['metadata']['preprocessing']['language_check']}")
    logger.info(f"Response message: {result['processed_message']}")
    
    # Test French
    test_message = "Comment puis-je trouver une clinique d'avortement près de chez moi?"
    result = preprocessor.process(test_message)
    logger.info(f"French test: {result['is_processable']} - {result['metadata']['preprocessing']['language_check']}")
    
    # Test very short message
    test_message = "Hi"
    result = preprocessor.process(test_message)
    logger.info(f"Short message test: {result['is_processable']} - {result['metadata']['preprocessing']['language_check']}")

def test_pii_redaction():
    """Test the PII redaction functionality"""
    logger.info("\n=== Testing PII Redaction ===")
    
    preprocessor = Preprocessor()
    
    # Test email redaction
    test_message = "Please contact me at test.user@example.com for more information."
    result = preprocessor.process(test_message)
    logger.info(f"Email redaction: {result['processed_message']}")
    logger.info(f"Redaction info: {result['metadata']['preprocessing']['pii_redaction']}")
    
    # Test phone redaction
    test_message = "My phone number is (555) 123-4567, call me anytime."
    result = preprocessor.process(test_message)
    logger.info(f"Phone redaction: {result['processed_message']}")
    logger.info(f"Redaction info: {result['metadata']['preprocessing']['pii_redaction']}")
    
    # Test different phone formats
    test_message = "Contact numbers: 555-123-4567, +1 555 123 4567, 5551234567"
    result = preprocessor.process(test_message)
    logger.info(f"Multiple phone formats: {result['processed_message']}")
    
    # Test both email and phone
    test_message = "Contact me at user@example.org or call 555-987-6543."
    result = preprocessor.process(test_message)
    logger.info(f"Email and phone: {result['processed_message']}")
    
    # Test ZIP code preservation (should NOT be redacted)
    test_message = "I live in ZIP code 90210 and need information about laws."
    result = preprocessor.process(test_message)
    logger.info(f"ZIP preservation: {result['processed_message']}")
    
    # Test email within URL (should NOT be redacted since it's not a real email)
    test_message = "Visit https://www.example.com/path@something.html for more info."
    result = preprocessor.process(test_message)
    logger.info(f"URL with @ symbol: {result['processed_message']}")

def test_typo_correction():
    """Test the typo correction functionality"""
    logger.info("\n=== Testing Typo Correction ===")
    
    preprocessor = Preprocessor()
    
    # Test simple typo
    test_message = "Where can I get an aborton?"
    result = preprocessor.process(test_message)
    logger.info(f"Simple typo: '{test_message}' -> '{result['processed_message']}'")
    logger.info(f"Correction info: {result['metadata']['preprocessing']['typo_correction']}")
    
    # Test multiple typos
    test_message = "What are my contraceptve optins?"
    result = preprocessor.process(test_message)
    logger.info(f"Multiple typos: '{test_message}' -> '{result['processed_message']}'")
    
    # Test long message (correction should be skipped)
    test_message = "I need to know about aborton laws in my state but I'm writing a very long message with lots of words so that the system doesn't correct the typo in this case because it's longer than 10 words."
    result = preprocessor.process(test_message)
    logger.info(f"Long message typo correction: {result['metadata']['preprocessing']['typo_correction']}")
    logger.info(f"Message unchanged: {test_message == result['processed_message']}")

def test_zip_lookup():
    """Test the ZIP code lookup functionality"""
    logger.info("\n=== Testing ZIP Code Lookup ===")
    
    preprocessor = Preprocessor()
    
    # Test valid ZIP codes across different states
    test_zips = {
        "90210": "CA",  # Beverly Hills, CA
        "10001": "NY",  # New York, NY
        "60611": "IL",  # Chicago, IL
        "77001": "TX",  # Houston, TX
        "02108": "MA",  # Boston, MA
    }
    
    for zip_code, expected_state in test_zips.items():
        state = preprocessor.get_state_from_zip(zip_code)
        logger.info(f"ZIP {zip_code} -> State: {state} (Expected: {expected_state})")
    
    # Test invalid ZIP
    state = preprocessor.get_state_from_zip("00000")
    logger.info(f"Invalid ZIP 00000 -> State: {state} (Expected: None)")
    
    # Test ZIP embedded in text
    state = preprocessor.get_state_from_zip("My ZIP is 90210, can you help?")
    logger.info(f"ZIP in text -> State: {state} (Expected: CA)")

def test_complete_flow():
    """Test a complete preprocessing flow with all steps"""
    logger.info("\n=== Testing Complete Preprocessing Flow ===")
    
    preprocessor = Preprocessor()
    
    # Test with everything
    test_message = "I live in 90210 and my email is user@example.com. I need info about aborton laws."
    result = preprocessor.process(test_message)
    
    logger.info(f"Original message: {test_message}")
    logger.info(f"Processed message: {result['processed_message']}")
    logger.info(f"Preprocessing metadata: {json.dumps(result['metadata']['preprocessing'], indent=2)}")

def main():
    """Run all tests"""
    logger.info("Starting Preprocessor tests...")
    
    try:
        test_language_detection()
        test_pii_redaction()
        test_typo_correction()
        test_zip_lookup()
        test_complete_flow()
        
        logger.info("\n=== All tests completed! ===")
    except Exception as e:
        logger.error(f"Error during tests: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 