import logging
import re
import os
from typing import Dict, Optional, Tuple, Union, Any
try:
    import zipcodes
    HAVE_ZIPCODES = True
except ImportError:
    HAVE_ZIPCODES = False
    
from symspellpy import SymSpell, Verbosity
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Preprocessor module for the chatbot pipeline.
    
    This class handles input validation and cleaning before message processing:
    1. Language Detection (English only)
    2. PII Redaction (email, phone numbers)
    3. Typo Correction for short queries
    4. ZIP Code Lookup support
    """
    
    def __init__(self):
        """Initialize the preprocessor with required components"""
        logger.info("Initializing Preprocessor")
        
        # Initialize SymSpell for typo correction
        self.sym_spell = None
        self.initialize_symspell()
        
        # Check if zipcodes library is available
        if HAVE_ZIPCODES:
            logger.info("Successfully initialized zipcodes library")
        else:
            logger.warning("zipcodes library not available, will use fallback ZIP lookup")
            
        # PII detection patterns
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b')
        self.zip_pattern = re.compile(r'\b(\d{5})\b')
        
        # List of US states to preserve during typo correction
        self.us_states = [
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
        self.state_variants = {
            "californie": "california",  # French
            "californië": "california",  # Dutch
            "tejas": "texas",           # Spanish variant
            "nuevo mexico": "new mexico", # Spanish
            "nueva york": "new york",   # Spanish
            "floride": "florida",       # French
        }
        
        self.state_abbreviations = [
            "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", 
            "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", 
            "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", 
            "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", 
            "wi", "wy", "dc"
        ]
        
        # State abbreviations that need special handling to avoid false positives
        self.ambiguous_abbrs = {
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
        
        # Simple ZIP code to state mapping for fallback
        self.zip_ranges = {
            'AL': (35000, 36999),
            'AK': (99500, 99999),
            'AZ': (85000, 86999),
            'AR': (71600, 72999),
            'CA': (90000, 96699),
            'CO': (80000, 81999),
            'CT': (6000, 6999),
            'DE': (19700, 19999),
            'DC': (20000, 20599),
            'FL': (32000, 34999),
            'GA': (30000, 31999),
            'HI': (96700, 96899),
            'ID': (83200, 83999),
            'IL': (60000, 62999),
            'IN': (46000, 47999),
            'IA': (50000, 52999),
            'KS': (66000, 67999),
            'KY': (40000, 42799),
            'LA': (70000, 71599),
            'ME': (3900, 4999),
            'MD': (20600, 21999),
            'MA': (1000, 2799),
            'MI': (48000, 49999),
            'MN': (55000, 56999),
            'MS': (38600, 39999),
            'MO': (63000, 65999),
            'MT': (59000, 59999),
            'NE': (68000, 69999),
            'NV': (89000, 89999),
            'NH': (3000, 3899),
            'NJ': (7000, 8999),
            'NM': (87000, 88499),
            'NY': (10000, 14999),
            'NC': (27000, 28999),
            'ND': (58000, 58999),
            'OH': (43000, 45999),
            'OK': (73000, 74999),
            'OR': (97000, 97999),
            'PA': (15000, 19699),
            'RI': (2800, 2999),
            'SC': (29000, 29999),
            'SD': (57000, 57999),
            'TN': (37000, 38599),
            'TX': (75000, 79999),
            'UT': (84000, 84999),
            'VT': (5000, 5999),
            'VA': (22000, 24699),
            'WA': (98000, 99499),
            'WV': (24700, 26999),
            'WI': (53000, 54999),
            'WY': (82000, 83199)
        }
    
    def initialize_symspell(self):
        """Initialize SymSpell dictionary for typo correction"""
        try:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            
            # Try to find dictionary path
            dictionary_path = None
            possible_paths = [
                "frequency_dictionary_en_82_765.txt",  # Current directory
                os.path.join("data", "frequency_dictionary_en_82_765.txt"),  # Data directory
                os.path.join("models", "frequency_dictionary_en_82_765.txt"),  # Models directory
                os.path.join("chatbot", "data", "frequency_dictionary_en_82_765.txt"),  # Chatbot data directory
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    dictionary_path = path
                    break
            
            if dictionary_path:
                term_index = 0  # Column of the term in the dictionary file
                count_index = 1  # Column of the term frequency in the dictionary file
                if not self.sym_spell.load_dictionary(dictionary_path, term_index, count_index):
                    logger.warning(f"Dictionary found at {dictionary_path} but failed to load")
                    self.create_basic_dictionary()
                else:
                    logger.info(f"SymSpell dictionary loaded from {dictionary_path}")
            else:
                logger.warning("No dictionary file found, creating basic dictionary")
                self.create_basic_dictionary()
                
        except Exception as e:
            logger.error(f"Error initializing SymSpell: {str(e)}")
            self.sym_spell = None
    
    def create_basic_dictionary(self):
        """Create a basic dictionary with common abortion-related terms"""
        if not self.sym_spell:
            return
            
        # Add common terms related to abortion, healthcare, etc.
        common_terms = {
            "abortion": 10000,
            "pregnant": 9000,
            "pregnancy": 9000,
            "contraception": 8000,
            "contraceptive": 7500,
            "pill": 7000,
            "emergency": 6500,
            "policy": 6000,
            "legal": 5500,
            "illegal": 5500,
            "law": 5000,
            "laws": 5000,
            "clinic": 4500,
            "doctor": 4000,
            "medical": 3500,
            "surgical": 3000,
            "medication": 3000,
            "weeks": 2500,
            "months": 2500,
            "trimester": 2000,
            "procedure": 2000,
            "options": 1500,
            "support": 1500,
            "help": 1000,
        }
        
        for term, count in common_terms.items():
            self.sym_spell.create_dictionary_entry(term, count)
            
        logger.info("Created basic dictionary with common abortion-related terms")
    
    def process(self, message: str) -> Dict[str, Any]:
        """
        Process an incoming message through all preprocessing steps
        
        Args:
            message (str): The raw user message
            
        Returns:
            Dict[str, Any]: Processing result with:
                - processed_message: The processed message text
                - is_processable: Whether the message can be processed (e.g., English)
                - stop_reason: Reason for stopping if not processable
                - metadata: Processing metadata
        """
        result = {
            "original_message": message,
            "processed_message": message,
            "is_processable": True,
            "stop_reason": None,
            "metadata": {
                "preprocessing": {
                    "language_check": "not_performed",
                    "pii_redaction": "not_performed",
                    "typo_correction": "not_performed",
                }
            }
        }
        
        # 1. Language Detection
        lang_result = self._check_language(message)
        result["metadata"]["preprocessing"]["language_check"] = lang_result
        
        if lang_result not in ["english", "language_detection_disabled"]:
            result["is_processable"] = False
            result["stop_reason"] = "non_english_message"
            return result
        
        # 2. PII Redaction
        redacted_message, redaction_info = self._redact_pii(message)
        result["metadata"]["preprocessing"]["pii_redaction"] = redaction_info
        result["processed_message"] = redacted_message
        
        # 3. Apply typo correction (which now always returns the original message)
        corrected_message, correction_info = self._correct_typos(redacted_message)
        result["metadata"]["preprocessing"]["typo_correction"] = correction_info
        result["processed_message"] = corrected_message
        
        return result
    
    def _check_language(self, message: str) -> str:
        """
        Check if the message is in English
        
        Args:
            message (str): Message to check
            
        Returns:
            str: Language detection result code
        """
        # Skip language detection entirely - always return English
        logger.info("Language detection disabled - treating all messages as English")
        return "language_detection_disabled"
    
    def _redact_pii(self, message: str) -> Tuple[str, str]:
        """
        Redact PII like emails and phone numbers from message
        
        Args:
            message (str): Message to process
            
        Returns:
            Tuple of (redacted_message, redaction_info)
        """
        redaction_info = "no_pii_found"
        redacted = message
        
        # Redact email addresses
        email_matches = self.email_pattern.findall(message)
        if email_matches:
            redaction_info = f"redacted_{len(email_matches)}_emails"
            for email in email_matches:
                redacted = redacted.replace(email, "[EMAIL REDACTED]")
            logger.info(f"Redacted {len(email_matches)} email(s) from message")
        
        # Redact phone numbers
        phone_matches = self.phone_pattern.findall(redacted)
        if phone_matches:
            if redaction_info == "no_pii_found":
                redaction_info = f"redacted_{len(phone_matches)}_phones"
            else:
                redaction_info += f"_and_{len(phone_matches)}_phones"
                
            # Get full matches from the pattern
            all_matches = []
            for match in self.phone_pattern.finditer(redacted):
                all_matches.append(match.group(0))
                
            for phone in all_matches:
                redacted = redacted.replace(phone, "[PHONE REDACTED]")
            logger.info(f"Redacted {len(phone_matches)} phone number(s) from message")
        
        # Preserve ZIP codes intentionally - do not redact
        
        return redacted, redaction_info
    
    def _correct_typos(self, message: str) -> Tuple[str, str]:
        """
        Correct typos in short messages
        
        Args:
            message (str): Message to correct
            
        Returns:
            Tuple of (corrected_message, correction_info)
        """
        # Skip all typo correction - always return the original message
        logger.info("Typo correction completely disabled - returning original message")
        return message, "typo_correction_disabled"
    
    def get_state_from_zip(self, zip_code: str) -> Optional[str]:
        """
        Get state code from ZIP code
        
        Args:
            zip_code (str): 5-digit ZIP code
            
        Returns:
            Optional[str]: State code or None if not found
        """
        try:
            # Handle string or match object
            if not isinstance(zip_code, str):
                zip_code = str(zip_code)
                
            # Extract 5-digit ZIP code if embedded in text
            zip_match = self.zip_pattern.search(zip_code)
            if zip_match:
                zip_code = zip_match.group(1)
            
            # Try using the zipcodes library if available
            if HAVE_ZIPCODES:
                if zipcodes.is_real(zip_code):
                    # Get the ZIP code data
                    zip_data = zipcodes.matching(zip_code)
                    if zip_data and len(zip_data) > 0 and 'state' in zip_data[0]:
                        state = zip_data[0]['state']
                        logger.info(f"Matched ZIP code {zip_code} to state {state} using zipcodes library")
                        return state
            
            # Fallback to the ZIP ranges if needed
            zip_int = int(zip_code)
            for state, (lower, upper) in self.zip_ranges.items():
                if lower <= zip_int <= upper:
                    logger.info(f"Matched ZIP code {zip_code} to state {state} using fallback ranges")
                    return state
                    
            logger.warning(f"ZIP code {zip_code} found but no state matched")
            return None
                
        except Exception as e:
            logger.error(f"Error in ZIP code lookup: {str(e)}")
            return None 