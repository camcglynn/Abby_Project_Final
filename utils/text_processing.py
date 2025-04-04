import re
import logging
from langdetect import detect, LangDetectException
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# List of US states to preserve during language detection
US_STATES = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", 
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", 
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", 
    "maine", "maryland", "massachusetts", "michigan", "minnesota", 
    "mississippi", "missouri", "montana", "nebraska", "nevada", 
    "new hampshire", "new jersey", "new mexico", "new york", 
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon", 
    "pennsylvania", "rhode island", "south carolina", "south dakota", 
    "tennessee", "texas", "utah", "vermont", "virginia", "washington", 
    "west virginia", "wisconsin", "wyoming", "district of columbia", "d.c."
]

# Common foreign language variants of state names
STATE_VARIANTS = {
    "californie": "california",  # French
    "californië": "california",  # Dutch
    "tejas": "texas",           # Spanish variant
    "nuevo mexico": "new mexico", # Spanish
    "nueva york": "new york",   # Spanish
    "floride": "florida",       # French
}

# State abbreviations that need special handling to avoid false positives
AMBIGUOUS_ABBRS = {
    "in": "Indiana",  # Common preposition
    "me": "Maine",    # Personal pronoun
    "or": "Oregon",   # Conjunction
    "de": "Delaware", # Common in many languages as preposition
    "la": "Louisiana", # Spanish article/French article
    "pa": "Pennsylvania", # Spanish stop word
    "hi": "Hawaii",   # Greeting
    "oh": "Ohio",     # Exclamation
    "ok": "Oklahoma", # Affirmation 
    "va": "Virginia", # Goes (Spanish)
    "id": "Idaho",    # Identity
    "ma": "Massachusetts", # Possessive (French)
    "mo": "Missouri", # Moment (Spanish)
    "al": "Alabama",  # Spanish preposition
    "ut": "Utah",     # Latin word
    "ct": "Connecticut", # Common abbreviation
    "ri": "Rhode Island", # Spanish word
}

STATE_ABBREVIATIONS = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", 
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", 
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", 
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", 
    "wi", "wy", "dc"
]

def clean_text(text):
    """
    Clean text by removing extra whitespace, normalizing, etc.
    
    Args:
        text (str): Text to clean
    
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def detect_language(text):
    """
    Detect the language of the given text
    
    Args:
        text (str): Text to analyze
    
    Returns:
        str: Language code (e.g., 'en' for English)
    """
    # Skip language detection entirely - always return English
    logger.info("Language detection disabled - treating all messages as English")
    return 'en'  # Always return English

def extract_keywords(text, max_keywords=5):
    """
    Extract important keywords from text
    
    Args:
        text (str): Text to analyze
        max_keywords (int): Maximum number of keywords to extract
    
    Returns:
        list: Extracted keywords
    """
    # This is a simple keyword extraction based on word frequency
    # In a real implementation, you might use NLTK, spaCy, or another NLP library
    
    if not text:
        return []
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stopwords
    stopwords = {'the', 'and', 'or', 'but', 'because', 'as', 'what', 'when', 
                'where', 'how', 'why', 'who', 'which', 'this', 'that', 'these', 
                'those', 'with', 'for', 'from', 'to', 'at', 'by', 'about', 'like'}
    filtered_words = [word for word in words if word not in stopwords]
    
    # Count word frequency
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in keywords[:max_keywords]]

def is_question(text):
    """
    Check if text is likely a question
    
    Args:
        text (str): Text to analyze
    
    Returns:
        bool: True if text is likely a question, False otherwise
    """
    # Simple heuristics to detect questions
    # More sophisticated analysis could be used in a real implementation
    
    text = text.strip()
    
    # Check for question marks
    if '?' in text:
        return True
    
    # Check for question words at the beginning
    question_starters = ['what', 'who', 'when', 'where', 'why', 'how', 'is', 'are', 
                         'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should']
    first_word = text.lower().split()[0] if text else ''
    
    return first_word in question_starters

class PIIDetector:
    """
    Detects and redacts Personally Identifiable Information (PII) in text
    """
    def __init__(self):
        """Initialize PII detection patterns"""
        # Patterns for various PII types
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'address': r'\b\d+\s+([A-Za-z]+\s+){1,3}(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Plaza|Plz|Terrace|Ter|Place|Pl)\b',
            'zip_code': r'\b\d{5}(?:[-\s]\d{4})?\b',
            'date_of_birth': r'\b(0[1-9]|1[0-2])[-/.](0[1-9]|[12][0-9]|3[01])[-/.](19|20)\d{2}\b'
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in self.patterns.items()}
        
        logger.info("PII Detector initialized")
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text
        
        Args:
            text (str): Text to check for PII
            
        Returns:
            List[Dict]: List of detected PII items with type and location
        """
        pii_items = []
        
        for pii_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                pii_items.append({
                    'type': pii_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return pii_items
    
    def has_pii(self, text: str) -> bool:
        """
        Check if text contains any PII
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if PII is detected
        """
        return any(pattern.search(text) for pattern in self.compiled_patterns.values())
    
    def redact_pii(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Redact PII from text
        
        Args:
            text (str): Text containing PII
            
        Returns:
            Tuple[str, List]: Redacted text and list of redacted items
        """
        redacted_text = text
        detected_items = self.detect_pii(text)
        
        # Sort in reverse order to maintain correct indices when replacing
        detected_items.sort(key=lambda x: x['start'], reverse=True)
        
        for item in detected_items:
            pii_type = item['type']
            start = item['start']
            end = item['end']
            
            # Create redaction placeholder based on type
            if pii_type == 'email':
                redaction = '[EMAIL REDACTED]'
            elif pii_type == 'phone':
                redaction = '[PHONE REDACTED]'
            elif pii_type == 'ssn':
                redaction = '[SSN REDACTED]'
            elif pii_type == 'credit_card':
                redaction = '[CREDIT CARD REDACTED]'
            elif pii_type == 'address':
                redaction = '[ADDRESS REDACTED]'
            elif pii_type == 'zip_code':
                redaction = '[ZIP CODE REDACTED]'
            elif pii_type == 'date_of_birth':
                redaction = '[DOB REDACTED]'
            else:
                redaction = '[REDACTED]'
            
            # Replace the PII with redaction
            redacted_text = redacted_text[:start] + redaction + redacted_text[end:]
        
        return redacted_text, detected_items
    
    def warn_about_pii(self, text: str) -> Optional[str]:
        """
        Generate warning message if PII is detected
        
        Args:
            text (str): Text to check
            
        Returns:
            Optional[str]: Warning message or None if no PII detected
        """
        detected_items = self.detect_pii(text)
        
        if not detected_items:
            return None
            
        # Count PII by type
        pii_types = {}
        for item in detected_items:
            pii_type = item['type']
            pii_types[pii_type] = pii_types.get(pii_type, 0) + 1
        
        # Create warning message
        warning = "I noticed you may have shared personal information in your message. "
        warning += "For your privacy and security, please avoid sharing "
        
        pii_list = []
        for pii_type, count in pii_types.items():
            pii_list.append(f"{pii_type.replace('_', ' ')}")
        
        if len(pii_list) == 1:
            warning += f"{pii_list[0]}."
        elif len(pii_list) == 2:
            warning += f"{pii_list[0]} or {pii_list[1]}."
        else:
            warning += ", ".join(pii_list[:-1]) + f", or {pii_list[-1]}."
        
        warning += " This information will not be stored, but it's best practice to keep it private."
        
        return warning
        
    def detect_and_sanitize(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Detect PII in text and return sanitized version along with a warning
        
        Args:
            text (str): Text to check for PII
            
        Returns:
            Tuple[str, Optional[str]]: (Sanitized text, Warning message or None)
        """
        if not self.has_pii(text):
            return text, None
            
        # Sanitize the text by redacting PII
        sanitized_text, _ = self.redact_pii(text)
        
        # Generate appropriate warning
        warning = self.warn_about_pii(text)
        
        return sanitized_text, warning
