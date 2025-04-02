"""
Question classifier using GPT-4o to intelligently categorize user questions
for the reproductive health chatbot.
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class QuestionClassifier:
    """
    Uses GPT-4o to classify questions into different categories
    for appropriate handling in the reproductive health chatbot.
    """
    
    def __init__(self):
        """Initialize the question classifier with OpenAI client"""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        # Check if OpenAI is available
        try:
            from openai import OpenAI
            if self.openai_api_key:
                self.client = OpenAI(api_key=self.openai_api_key)
                self.openai_available = True
                logger.info("Question Classifier initialized with GPT-4o preference (will fallback to GPT-3.5-turbo if needed)")
            else:
                logger.warning("OPENAI_API_KEY not found, using rule-based classification")
                self.client = None
                self.openai_available = False
        except ImportError:
            logger.warning("OpenAI package not installed, using rule-based classification")
            self.client = None
            self.openai_available = False
        
    def extract_location(self, message: str) -> Optional[str]:
        """
        Extract location mentions (primarily US states) from the message
        
        Args:
            message (str): The user's message
            
        Returns:
            Optional[str]: The extracted location or None
        """
        # US States dictionary (full names and abbreviations)
        us_states = {
            "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
            "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
            "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
            "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
            "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
            "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
            "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
            "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
            "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
            "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
            "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
            "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
            "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC", "dc": "DC",
            "washington dc": "DC"
        }
        
        # Also recognize state abbreviations
        state_abbrs = {v.lower(): k for k, v in us_states.items()}
        
        # Pattern to look for location context in the message
        location_prefixes = [
            "in", "for", "about", "regarding", "at", "from", "to", "of"
        ]
        message_lower = message.lower()
        
        # First check for direct state mention
        for state in us_states:
            if f" {state} " in f" {message_lower} " or message_lower.startswith(f"{state} ") or message_lower.endswith(f" {state}"):
                return state
                
        # Check for state abbreviations
        for abbr in state_abbrs:
            # Make sure we're matching whole words, not parts of words
            pattern = r'\b' + re.escape(abbr) + r'\b'
            if re.search(pattern, message_lower):
                return state_abbrs[abbr]
        
        # Check for more complex location patterns
        for prefix in location_prefixes:
            pattern = fr'\b{prefix}\s+([A-Z][a-z]+(\s+[A-Z][a-z]+)*)\b'
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                potential_location = match.group(1).lower()
                if potential_location in us_states:
                    return potential_location
        
        return None
        
    def classify_question(self, question: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Classify a user question into appropriate categories using GPT-4o
        
        Args:
            question (str): The user's question
            history (Optional[List[Dict[str, Any]]]): Conversation history
            
        Returns:
            Dict[str, Any]: Classification results including categories, confidence, etc.
        """
        try:
            # Extract location first using regex for efficiency
            location_context = self.extract_location(question)
            
            # Check if the question is just a simple state name
            if len(question.split()) == 1 and location_context:
                logger.info(f"Question is just a state name: {location_context}")
                return {
                    "categories": ["location_only"],
                    "is_abortion_related": True,  # Assume it's related to previous abortion questions
                    "is_policy_question": True,  # Likely a follow-up to policy question
                    "is_emotional": False,
                    "location_context": location_context,
                    "confidence": 0.95
                }
            
            # If OpenAI is not available, use rule-based classification
            if not self.openai_available or not self.client:
                logger.warning("OpenAI not available, using rule-based classification")
                return self._rule_based_classification(question, history, location_context)
                
            # Define the prompt for GPT-4o classification
            system_message = """You are a specialized classifier for reproductive health questions. 
Your task is to categorize user questions to help route them to the right response system.
Analyze the question and return ONLY a JSON object with the following fields:
- categories: array of categories from ["policy", "information", "emotional_support", "greeting", "other"]
- is_abortion_related: boolean indicating if the question is related to abortion
- is_policy_question: boolean indicating if the question is specifically asking about abortion laws, regulations, or access
- is_emotional: boolean indicating if the question contains emotional content
- confidence: number between 0 and 1 indicating your confidence in this classification

Do not include any text outside the JSON object. The JSON should be valid and parseable."""

            # Include relevant conversation history
            context = ""
            if history and len(history) > 0:
                last_exchanges = history[-3:] if len(history) >= 3 else history
                context = "Previous conversation context:\n"
                for entry in last_exchanges:
                    context += f"{entry['sender']}: {entry['message']}\n"
                context += "\n"
            
            # Make the OpenAI API call
            try:
                from openai import OpenAI
                try:
                    # First try with GPT-4o
                    response = self.client.chat.completions.create(
                        model="gpt-4o",  # Using GPT-4o for better classification accuracy
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": f"{context}User question: {question}"}
                        ],
                        temperature=0.1,  # Low temperature for more consistent results
                        max_tokens=150
                    )
                    logger.info("Successfully used GPT-4o for classification")
                except Exception as model_error:
                    # Fall back to GPT-3.5-turbo if GPT-4o is not available
                    logger.warning(f"GPT-4o not available: {str(model_error)}. Falling back to GPT-3.5-turbo")
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",  # Fallback to GPT-3.5-turbo
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": f"{context}User question: {question}"}
                        ],
                        temperature=0.1,
                        max_tokens=150
                    )
                
                # Extract and parse the JSON response
                result_text = response.choices[0].message.content.strip()
                classification = json.loads(result_text)
                
                # Add the location context from our regex extraction
                classification["location_context"] = location_context
                
                # Log the classification result
                logger.info(f"Question classified as: {classification}")
                return classification
            except Exception as openai_error:
                logger.error(f"Error with OpenAI API: {str(openai_error)}")
                # Fall back to rule-based classification
                return self._rule_based_classification(question, history, location_context)
            
        except Exception as e:
            logger.error(f"Error classifying question: {str(e)}", exc_info=True)
            # Extract location again to ensure it's defined in the error path
            try:
                location_context = self.extract_location(question)
            except Exception:
                location_context = None
                
            # Fallback classification - very simple
            return {
                "categories": ["information"],
                "is_abortion_related": "abortion" in question.lower(),
                "is_policy_question": False,
                "is_emotional": False,
                "location_context": location_context,
                "confidence": 0.5
            }
    
    def is_abortion_policy_question(self, question: str, history: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, Optional[str]]:
        """
        Determine if a question is specifically about abortion policy or access
        
        Args:
            question (str): The user's question
            history (Optional[List[Dict[str, Any]]]): Conversation history
            
        Returns:
            Tuple[bool, Optional[str]]: (is_policy_question, location_context)
        """
        classification = self.classify_question(question, history)
        return (
            classification["is_policy_question"] and classification["is_abortion_related"],
            classification["location_context"]
        )
    
    def is_emotional_query(self, question: str) -> bool:
        """
        Determine if a question contains emotional content
        
        Args:
            question (str): The user's question
            
        Returns:
            bool: True if the question has emotional content
        """
        classification = self.classify_question(question)
        return classification["is_emotional"]
        
    def get_full_classification(self, question: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Get a complete classification of a question for advanced routing
        
        Args:
            question (str): The user's question
            history (Optional[List[Dict[str, Any]]]): Conversation history
            
        Returns:
            Dict[str, Any]: Complete classification details
        """
        return self.classify_question(question, history)
        
    def _rule_based_classification(self, question: str, history: Optional[List[Dict[str, Any]]] = None, location_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform rule-based classification when OpenAI is not available
        
        Args:
            question (str): The user's question
            history (Optional[List]): Conversation history
            location_context (Optional[str]): Pre-extracted location context
            
        Returns:
            Dict[str, Any]: Classification results
        """
        question_lower = question.lower()
        
        # Default classification
        classification = {
            "categories": ["information"],
            "is_abortion_related": False,
            "is_policy_question": False,
            "is_emotional": False,
            "location_context": location_context,
            "confidence": 0.7
        }
        
        # Check for abortion-related terms
        abortion_terms = ["abortion", "terminate", "termination", "roe v wade", "pro-choice", "pro-life"]
        classification["is_abortion_related"] = any(term in question_lower for term in abortion_terms)
        
        # Check for policy-related terms
        policy_terms = ["law", "legal", "illegal", "allowed", "banned", "policy", "right", "access",
                        "restrictions", "permit", "permission", "mandate", "require", "regulation"]
        contains_policy_terms = any(term in question_lower for term in policy_terms)
        
        # Check for emotional content
        emotional_terms = ["scared", "afraid", "worried", "anxious", "nervous", "terrified",
                          "concerned", "fear", "scary", "upset", "sad", "depressed", "frightened",
                          "help me", "desperate", "alone", "lonely", "cry", "crying", "tears"]
        classification["is_emotional"] = any(term in question_lower for term in emotional_terms)
        
        # Check for state mention (via location_context or in the question)
        has_location = bool(location_context) or any(state in question_lower for state in 
                        ["alabama", "alaska", "arizona", "arkansas", "california", "colorado",
                         "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", 
                         "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", 
                         "maine", "maryland", "massachusetts", "michigan", "minnesota", 
                         "mississippi", "missouri", "montana", "nebraska", "nevada", 
                         "new hampshire", "new jersey", "new mexico", "new york", 
                         "north carolina", "north dakota", "ohio", "oklahoma", "oregon", 
                         "pennsylvania", "rhode island", "south carolina", "south dakota", 
                         "tennessee", "texas", "utah", "vermont", "virginia", "washington", 
                         "west virginia", "wisconsin", "wyoming", "district of columbia", "dc"])
        
        # Check for greetings
        greeting_terms = ["hello", "hi ", "hey", "greetings", "good morning", "good afternoon", 
                         "good evening", "what's up", "how are you", "nice to meet", "pleased to meet"]
        is_greeting = any(term in question_lower for term in greeting_terms)
        
        # Determine primary category
        if is_greeting:
            classification["categories"] = ["greeting"]
        elif classification["is_abortion_related"] and (contains_policy_terms or has_location):
            classification["categories"] = ["policy"]
            classification["is_policy_question"] = True
        elif classification["is_emotional"]:
            classification["categories"] = ["emotional_support"]
        elif any(term in question_lower for term in ["what", "how", "when", "where", "why", "who", "which"]):
            classification["categories"] = ["information"]
        else:
            # Default to information for anything else
            classification["categories"] = ["information"]
            
        # State-only input special case
        if len(question.split()) == 1 and location_context:
            # Check history for abortion context
            if history and any("abortion" in msg.get("message", "").lower() 
                               for msg in history if msg.get("sender") == "user"):
                classification["categories"] = ["location_only"]
                classification["is_abortion_related"] = True
                classification["is_policy_question"] = True
                classification["confidence"] = 0.95
        
        return classification