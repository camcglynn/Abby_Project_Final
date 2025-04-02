#!/usr/bin/env python3
import logging
from utils.text_processing import detect_language
from chatbot.preprocessor import Preprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_detect_language():
    """Test the updated detect_language function with various inputs"""
    test_cases = [
        # English with state names
        ("I live in California", "en", "English with state name"),
        ("Texas has restrictive abortion laws", "en", "English with state name"),
        ("NY is a progressive state", "en", "English with state abbreviation"),
        
        # Spanish with state names
        ("Vivo en California y necesito ayuda", "es", "Spanish with state name"),
        ("Texas tiene leyes restrictivas sobre el aborto", "es", "Spanish with state name"),
        ("Acabo de mudarme a Florida y necesito información", "es", "Spanish with state name"),
        
        # French with state names
        ("Je vis en Californie et j'ai besoin d'aide", "fr", "French with state name"),
        ("Le Texas a des lois restrictives sur l'avortement", "fr", "French with state name"),
        
        # Non-state content in different languages
        ("¿Dónde puedo encontrar una clínica de aborto?", "es", "Spanish without state name"),
        ("J'ai besoin d'informations sur l'avortement", "fr", "French without state name"),
        ("I need information about abortion services", "en", "English without state name"),
    ]
    
    print("\n===== Testing detect_language function =====")
    for text, expected_lang, description in test_cases:
        detected = detect_language(text)
        result = "✅ PASS" if (detected == "en" and "state name" in description) else "ℹ️ INFO"
        print(f"{result} | {description}: \"{text}\" → Detected: {detected}, Expected base: {expected_lang}")

def test_preprocessor_language_check():
    """Test the Preprocessor._check_language method with various inputs"""
    preprocessor = Preprocessor()
    
    test_cases = [
        # English with state names
        ("I live in California", True, "English with state name"),
        ("Texas has restrictive abortion laws", True, "English with state name"),
        ("NY is a progressive state", True, "English with state abbreviation"),
        
        # Spanish with state names
        ("Vivo en California y necesito ayuda", True, "Spanish with state name"),
        ("Texas tiene leyes restrictivas sobre el aborto", True, "Spanish with state name"),
        ("Acabo de mudarme a Florida y necesito información", True, "Spanish with state name"),
        
        # French with state names
        ("Je vis en Californie et j'ai besoin d'aide", True, "French with state name"),
        ("Le Texas a des lois restrictives sur l'avortement", True, "French with state name"),
        
        # Non-state content in different languages
        ("¿Dónde puedo encontrar una clínica de aborto?", False, "Spanish without state name"),
        ("J'ai besoin d'informations sur l'avortement", False, "French without state name"),
        ("I need information about abortion services", True, "English without state name"),
    ]
    
    print("\n===== Testing Preprocessor._check_language method =====")
    for text, expected_is_english, description in test_cases:
        result = preprocessor._check_language(text)
        is_english = result["is_english"]
        status = result["status"]
        detected_lang = result["detected_language"]
        
        result_str = "✅ PASS" if is_english == expected_is_english else "❌ FAIL"
        print(f"{result_str} | {description}: \"{text}\" → is_english: {is_english}, status: {status}, detected: {detected_lang}")

if __name__ == "__main__":
    test_detect_language()
    test_preprocessor_language_check() 