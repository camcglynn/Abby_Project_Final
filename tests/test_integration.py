#!/usr/bin/env python
"""
Integration test script for Preprocessor with the chatbot pipeline.
"""

import logging
import sys
import asyncio
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("integration_test")

# Import the required classes
from chatbot import MultiAspectQueryProcessor
from chatbot import Preprocessor
from chatbot.policy_handler import PolicyHandler

async def test_integration():
    """Run tests to verify integration of Preprocessor with other components"""
    logger.info("Testing Preprocessor integration with chatbot components")
    
    # Test 1: Direct Preprocessor usage
    logger.info("\n=== Test 1: Direct Preprocessor usage ===")
    preprocessor = Preprocessor()
    result = preprocessor.process("I live in 90210 and need abortion information")
    logger.info(f"Preprocessor direct result: {result['metadata']['preprocessing']}")
    
    # Test 2: PolicyHandler ZIP code lookup via Preprocessor
    logger.info("\n=== Test 2: PolicyHandler ZIP code lookup ===")
    policy_handler = PolicyHandler()
    state = policy_handler._get_state_from_zip("90210")
    logger.info(f"PolicyHandler ZIP code lookup result: 90210 -> {state}")
    
    # Test 3: Full pipeline with MultiAspectQueryProcessor
    logger.info("\n=== Test 3: Full pipeline integration ===")
    processor = MultiAspectQueryProcessor()
    result = await processor.process_query("I live in 90210, what are the abortion laws here?")
    
    logger.info(f"Full pipeline result:")
    logger.info(f"- State code detected: {result.get('state_code', 'None')}")
    logger.info(f"- Preprocessing metadata included: {'preprocessing_metadata' in result}")
    logger.info(f"- Preprocessing data: {result.get('preprocessing_metadata', {})}")

    return 0

def main():
    """Main entry point"""
    try:
        asyncio.run(test_integration())
        logger.info("\nIntegration tests completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error during integration tests: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
