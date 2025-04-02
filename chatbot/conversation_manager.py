import logging
import time
import random
import uuid
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from chatbot.baseline_model import BaselineModel
from chatbot.friendly_bot import FriendlyBot
from chatbot.citation_manager import CitationManager
from chatbot.policy_api import PolicyAPI
from chatbot.question_classifier import QuestionClassifier
from chatbot.context_manager import ContextManager
import requests
import re
from utils.text_processing import PIIDetector, detect_language
import us
import openai
from flask import session
import nltk
from chatbot.response_evaluator import ResponseEvaluator
from utils.data_loader import load_reproductive_health_data

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages the conversation flow, integrating the baseline model with friendly elements
    """

    def __init__(self, evaluation_model="local"):
        """
        Initialize the conversation manager

        Args:
            evaluation_model (str): Model to use for response evaluation
                "openai": Use OpenAI's models only
                "local": Use local transformer models only
                "both": Use both (default)
        """
        logger.info(f"Initializing Conversation Manager with evaluation_model={evaluation_model}")
        try:
            # Initialize the baseline model with response evaluation capabilities
            self.baseline_model = BaselineModel(evaluation_model=evaluation_model)
            self.friendly_bot = FriendlyBot()
            self.citation_manager = CitationManager()
            self.policy_api = PolicyAPI()
            self.question_classifier = QuestionClassifier()
            self.context_manager = ContextManager(max_context_length=3)  # Initialize ContextManager
            self.conversation_history = []
            self._session_ended = False
            self.current_session_id = str(uuid.uuid4())  # Add session ID tracking
            
            # Initialize PII detector 
            self.pii_detector = PIIDetector()
            
            # Set up log file path
            self.log_dir = "logs"
            os.makedirs(self.log_dir, exist_ok=True)
            self.conversation_log_file = os.path.join(self.log_dir, "conversation_logs.json")
            
            # Initialize empty log file if it doesn't exist
            if not os.path.exists(self.conversation_log_file):
                with open(self.conversation_log_file, 'w') as f:
                    json.dump([], f)
                    
            # Add policy response cache to reduce API costs
            self.policy_cache = {}
            
            # Add general response cache
            self.response_cache = {}
            self.cache_ttl = 3600  # 1 hour in seconds
            
            # Set evaluation frequency to reduce costs (only evaluate every N messages)
            self.evaluation_frequency = int(os.environ.get("EVALUATION_FREQUENCY", "10"))
            self.message_counter = 0

            logger.info("Conversation Manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Conversation Manager: {str(e)}",
                         exc_info=True)
            raise

    def _save_conversation_logs(self):
        """Save conversation logs to file, ensuring all PII is properly redacted"""
        try:
            # Load existing logs
            try:
                with open(self.conversation_log_file, 'r') as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logs = []
            
            # Double-check for PII in history before saving to logs
            for message in self.conversation_history:
                # Only process user messages for PII
                if message['sender'] == 'user' and isinstance(message.get('message'), str):
                    # Check for PII once more before saving to disk
                    if self.pii_detector.has_pii(message['message']):
                        # Sanitize the message (redact PII) while preserving zip codes and state info
                        # Our updated PII detector will automatically preserve these items
                        sanitized_message, redacted_items = self.pii_detector.redact_pii(message['message'])
                        
                        # Update the message with sanitized version
                        message['message'] = sanitized_message
                        
                        # Record redaction for logging
                        if redacted_items:
                            pii_types = [item['type'] for item in redacted_items]
                            logger.info(f"Redacted {len(redacted_items)} PII items of types {pii_types} from message before saving to logs")
                
                # Convert timestamp to ISO format if it's a float
                if isinstance(message.get('timestamp'), float):
                    message['timestamp'] = datetime.fromtimestamp(message['timestamp']).isoformat()
                
                # Add to logs
                logs.append(message)
            
            # Save updated logs
            with open(self.conversation_log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
            # Reset conversation history
            self.conversation_history = []
            logger.debug(f"Saved {len(logs)} messages to conversation logs")
            
        except Exception as e:
            logger.error(f"Error saving conversation logs: {str(e)}")
            # Don't raise the exception - this is a background task

    def _extract_zip_code(self, message):
        """Extract zip code from message"""
        if not message:
            return None
            
        # Look for standard 5-digit US zip code
        zip_pattern = r'\b\d{5}\b'
        matches = re.findall(zip_pattern, message)
        if matches:
            zip_code = matches[0]
            logger.info(f"Extracted zip code: {zip_code}")
            return zip_code
            
        return None
        
    def process_message(self, message, conversation_id=None, persona="baseline"):
        """Process a message and generate a response"""
        message_id = str(uuid.uuid4())
        
        # Initialize conversation ID if not provided
        if conversation_id:
            self.current_session_id = conversation_id
            logger.info(f"Using existing conversation ID: {conversation_id}")
        else:
            self.current_session_id = str(uuid.uuid4())
            logger.info(f"Created new conversation ID: {self.current_session_id}")
        
        # Clean and preprocess the message
        message = self.clean_message(message)
        
        # Add message to history
        self.add_to_history('user', message)
        
        # Debug log the conversation history
        logger.info(f"Current conversation history (last 3): {self.conversation_history[-3:] if len(self.conversation_history) >= 3 else self.conversation_history}")
        
        # Check if this is a state-only message
        is_state_only = self.is_state_only_message(message)
        logger.info(f"Is state only message check result: {is_state_only}")
        
        # Check for abortion context either in current message or recent history
        has_abortion_context = self._is_abortion_related(message) or self._has_recent_abortion_context()
        logger.info(f"Has abortion context check result: {has_abortion_context}")
        
        # Force abortion context behavior for state-only messages when preceded by abortion query
        # This ensures that if we just got asked about abortion and then get a state name, we treat it as abortion policy
        if is_state_only and len(self.conversation_history) >= 2:
            prev_msgs = [entry.get('message', '').lower() for entry in self.conversation_history[-3:] if entry.get('sender') == 'bot']
            if any(('abortion' in msg or 'policy' in msg) and ('which state' in msg or 'what state' in msg or 'your state' in msg) for msg in prev_msgs):
                logger.info(f"Forcing abortion context for state-only message after abortion prompt")
                has_abortion_context = True
                is_state_only = 'abortion_context'
        
        if is_state_only == 'abortion_context' or (is_state_only and has_abortion_context):
            # Get state code
            state_code = None
            message_lower = message.lower()
            
            # Check direct match with state name
            for code, name in self.policy_api.STATE_NAMES.items():
                if message_lower == name.lower():
                    state_code = code
                    break
                
            # If no direct match, check for partial matches
            if not state_code:
                for code, name in self.policy_api.STATE_NAMES.items():
                    name_lower = name.lower()
                    if message_lower in name_lower or name_lower in message_lower:
                        # For multi-word states, verify first word matches
                        if message_lower.split() and name_lower.split() and message_lower.split()[0] == name_lower.split()[0]:
                            state_code = code
                            logger.info(f"Partial state name match: {message_lower} -> {name_lower}")
                            break
            
            # Check direct match with state abbreviation
            if not state_code and message_lower.upper() in self.policy_api.STATE_NAMES:
                state_code = message_lower.upper()
                
            if state_code:
                logger.info(f"Processing state-only message with abortion context for state: {state_code}")
                policy_response = self._get_policy_response_for_state(state_code, message_id)
                return policy_response
        
        # Check for policy context
        policy_context = self._is_policy_lookup(message) or has_abortion_context
        
        if policy_context:
            logger.info("Detected abortion policy lookup query or context")
            
            # Look for state name mention in the message
            state_name = self._check_for_state_names(message)
            logger.info(f"State mention identified: {state_name}")
            
            # Look for zip code in the message
            zip_code = self._extract_zip_code(message)
            logger.info(f"Zip code mention identified: {zip_code}")
            
            if state_name:
                # Handle case where state name is found
                state_code = state_name  # state_name from _check_for_state_names is actually the state code
                logger.info(f"State code mapped from name: {state_code}")
                if state_code:
                    response = self._get_policy_response_for_state(state_code, message_id)
                    return response
            elif zip_code:
                # Handle case where zip code is found
                state_code = self._get_state_from_zip_code(zip_code)
                logger.info(f"State code mapped from zip code: {state_code}")
                if state_code:
                    response = self._get_policy_response_for_state(state_code, message_id, zip_code=zip_code)
                    return response
            
            # No state or zip code found - ask for state
            response = {
                "text": "I'd be happy to provide information about abortion access and policies. Could you please let me know which state you're inquiring about? You can provide either the state name or your zip code.",
                "citations": [],
                "citation_objects": [],
                "message_id": message_id
            }
            
            # Add response to history
            self.add_to_history('bot', response['text'], message_id=message_id)
            return response
        
        try:
            # Process as normal message if no policy context
            response = self.baseline_model.process_message(message, conversation_id, persona)
            
            # Add response to history
            if response and 'text' in response:
                self.add_to_history('bot', response['text'], message_id=response.get('message_id', message_id))
            
            return response
        except Exception as e:
            logger.error(f"Error in baseline processing: {str(e)}")
            error_response = {
                "text": "I apologize, but I encountered an error processing your message. Could you please try rephrasing your question?",
                "citations": [],
                "citation_objects": [],
                "message_id": message_id
            }
            self.add_to_history('bot', error_response['text'], message_id=message_id)
            return error_response

    def _get_policy_response_for_state(self, state_code, message_id, zip_code=None):
        """Get formatted policy response for a state"""
        try:
            # Get policy data from API
            policy_data = self.policy_api.get_policy_data(state_code)
            state_name = self.policy_api.STATE_NAMES.get(state_code)
            
            # Format the response
            response_text = f"Here's what I know about abortion access in {state_name}:\n\n"
            
            # Add legal status
            if policy_data.get('legal_status'):
                response_text += f"**Legal Status**: {policy_data['legal_status']}\n\n"
            
            # Add gestational limits
            if policy_data.get('gestational_limit'):
                response_text += f"**Gestational Limit**: {policy_data['gestational_limit']}\n\n"
            
            # Add key restrictions
            if policy_data.get('restrictions'):
                response_text += "**Key Restrictions**:\n"
                for restriction in policy_data['restrictions']:
                    response_text += f"- {restriction}\n"
                response_text += "\n"
            
            # Add available services
            if policy_data.get('services'):
                response_text += "**Available Services**:\n"
                for service in policy_data['services']:
                    response_text += f"- {service}\n"
                response_text += "\n"
            
            # Add resources
            if policy_data.get('resources'):
                response_text += "**Resources**:\n"
                for resource in policy_data['resources']:
                    response_text += f"- {resource}\n"
                response_text += "\n"
            
            # Add disclaimer
            response_text += "\n*Note: This information is current as of my last update, but laws and policies can change. Please verify with healthcare providers or legal professionals for the most up-to-date information.*"
            
            # Format citations
            citations = []
            citation_objects = []
            if policy_data.get('sources'):
                for source in policy_data['sources']:
                    citation = {
                        'text': source.get('title', 'Policy Information'),
                        'url': source.get('url', ''),
                        'accessed_date': source.get('accessed_date', '')
                    }
                    citations.append(citation)
                    citation_objects.append({
                        'title': source.get('title', 'Policy Information'),
                        'url': source.get('url', ''),
                        'accessed_date': source.get('accessed_date', ''),
                        'source_type': 'policy'
                    })
            
            return {
                'text': response_text,
                'citations': citations,
                'citation_objects': citation_objects,
                'message_id': message_id
            }
            
        except Exception as e:
            logger.error(f"Error getting policy response for state {state_code}: {str(e)}")
            return {
                'text': f"I apologize, but I encountered an error while retrieving policy information for {state_name}. Please try again or ask about another state.",
                'citations': [],
                'citation_objects': [],
                'message_id': message_id
            }

    def _check_for_state_names(self, message):
        """
        Check for state names in a message
        
        Args:
            message (str): Message to check for state names
            
        Returns:
            str: State code or name if found, None otherwise
        """
        if not message:
            return None
            
        message_lower = message.lower().strip()
        
        # Handle single-word state queries directly (like "California")
        if message_lower in self.policy_api.STATE_NAMES_LOWER:
            state_code = self.policy_api.STATE_NAMES_LOWER[message_lower]
            logger.info(f"Found direct state name match: '{message_lower}' -> {state_code}")
            return state_code
            
        # Common words that are also state abbreviations to avoid false matches
        ambiguous_abbrs = {
            'in': 'Indiana',  # Skip 'in' as it's commonly used in phrases like "in my state"
            'me': 'Maine',
            'or': 'Oregon',
            'hi': 'Hawaii',
            'ok': 'Oklahoma',
            'oh': 'Ohio',
            'la': 'Louisiana',
            'wa': 'Washington',
            'pa': 'Pennsylvania',
            'ma': 'Massachusetts'
        }
        
        # Check if a substring is a whole word (surrounded by spaces or punctuation)
        def is_whole_word(text, substr):
            # First, escape special regex characters in substr
            escaped = re.escape(substr)
            # Create a pattern to match this substring as a whole word
            pattern = r'\b' + escaped + r'\b'
            # Check if it matches as a whole word
            return bool(re.search(pattern, text))
            
        # First check for full state names
        for state_code, state_name in self.policy_api.STATE_NAMES.items():
            if state_name.lower() in message_lower:
                word_positions = [m.start() for m in re.finditer(state_name.lower(), message_lower)]
                for pos in word_positions:
                    # Make sure it's a whole word match by checking nearby characters
                    if (pos == 0 or not message_lower[pos-1].isalpha()) and (pos + len(state_name) == len(message_lower) or not message_lower[pos + len(state_name)].isalpha()):
                        logger.info(f"Found state name '{state_name}' within message")
                        return state_code
                        
        # Then check for state abbreviations (only as whole words)
        for state_code in self.policy_api.STATE_NAMES.keys():
            # Skip ambiguous abbreviations like 'in' in common phrases
            if state_code.lower() in ambiguous_abbrs:
                # For ambiguous abbreviations, only match if the exact state name is mentioned,
                # not the abbreviation
                if ambiguous_abbrs[state_code.lower()].lower() in message_lower:
                    logger.info(f"Found state name '{ambiguous_abbrs[state_code.lower()]}' within message")
                    return state_code
                elif is_whole_word(message_lower, state_code.lower()) and not any(phrase in message_lower for phrase in ["in my state", "in your state", "abortion in"]):
                    # Only match 'in' if it's a whole word AND not part of these phrases
                    logger.info(f"Found state abbreviation '{state_code}' as whole word in message")
                    return state_code
            # For non-ambiguous abbreviations, look for whole word matches
            elif is_whole_word(message_lower, state_code.lower()):
                logger.info(f"Found state abbreviation '{state_code}' as whole word in message")
                return state_code
                
        # Special case for phrases like "abortion in Indiana" or "get an abortion in CA"
        for state_code, state_name in self.policy_api.STATE_NAMES.items():
            phrases = [
                f"abortion in {state_name.lower()}", 
                f"abortion in {state_code.lower()}",
                f"abortions in {state_name.lower()}",
                f"abortions in {state_code.lower()}"
            ]
            for phrase in phrases:
                if phrase in message_lower:
                    logger.info(f"Found state {state_code} mentioned in abortion context")
                    return state_code
                    
        return None

    def _is_zip_code(self, text):
        """Check if a string is a valid US zip code."""
        # Remove any non-numeric characters and check if it's a 5-digit number
        cleaned = ''.join(c for c in text if c.isdigit())
        return len(cleaned) == 5 and text.strip().isdigit()
        
    def _get_state_from_zip_code(self, zip_code):
        """
        Convert a zip code to a state code using an external API
        
        Args:
            zip_code (str): US ZIP code
            
        Returns:
            str: Two-letter state code or None if not found
        """
        if not zip_code or not re.match(r'^\d{5}$', zip_code):
            logger.warning(f"Invalid zip code format: {zip_code}")
            return None
            
        # Check if we're in testing/development mode with mock data
        if os.environ.get('TESTING') == 'true':
            # For testing, return mock state codes
            mock_states = {
                '90210': 'CA',
                '10001': 'NY',
                '60601': 'IL',
                '33101': 'FL',
                '75001': 'TX',
                '02108': 'MA',
                '20001': 'DC',
                '98101': 'WA',
                '48201': 'MI',
                '80201': 'CO'
            }
            return mock_states.get(zip_code)
            
        try:
            # First try the zipcodes library (most reliable)
            try:
                import zipcodes
                result = zipcodes.matching(zip_code)
                if result and len(result) > 0:
                    state_code = result[0]['state']
                    logger.info(f"Found state {state_code} for zip code {zip_code} using zipcodes library")
                    return state_code
            except (ImportError, Exception) as e:
                logger.warning(f"Zipcodes library unavailable or error: {str(e)}")
                
            # Try to get from Google Maps API next if zipcodes library fails
            api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
            if api_key:
                url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zip_code}&key={api_key}"
                response = requests.get(url, timeout=5)
                data = response.json()
                
                if data.get('status') == 'OK' and data.get('results'):
                    # Parse the components to find the state
                    for component in data['results'][0]['address_components']:
                        if 'administrative_area_level_1' in component['types']:
                            state_code = component['short_name']
                            logger.info(f"Found state {state_code} for zip code {zip_code} using Google Maps API")
                            return state_code
            
            # Fallback to ZipCodeAPI if Google fails
            api_key = os.environ.get('ZIPCODE_API_KEY')
            if api_key:
                url = f"https://www.zipcodeapi.com/rest/{api_key}/info.json/{zip_code}/degrees"
                response = requests.get(url, timeout=5)
                data = response.json()
                
                if 'state' in data:
                    state_code = data['state']
                    logger.info(f"Found state {state_code} for zip code {zip_code} using ZipCodeAPI")
                    return state_code
                
            # Final fallback to local basic mapping of ZIP code ranges
            # This is not comprehensive but covers the most common ranges
            zip_prefix = int(zip_code[:3])
            
            # Basic ZIP code prefix to state mapping (first 3 digits)
            state_ranges = {
                'AL': (350, 369),
                'AK': (995, 999),
                'AZ': (850, 865),
                'AR': (716, 729),
                'CA': [(900, 961), (940, 961)], # California has multiple ranges including San Jose (95xxx)
                'CO': (800, 816),
                'CT': ('060', '069'),
                'DE': (197, 199),
                'DC': (200, 205),
                'FL': (320, 349),
                'GA': (300, 319),
                'HI': (967, 968),
                'ID': (832, 839),
                'IL': (600, 629),
                'IN': (460, 479),
                'IA': (500, 528),
                'KS': (660, 679),
                'KY': (400, 427),
                'LA': (700, 714),
                'ME': ('039', '049'),
                'MD': (206, 219),
                'MA': ('010', '027'),
                'MI': (480, 499),
                'MN': (550, 567),
                'MS': (386, 397),
                'MO': (630, 658),
                'MT': (590, 599),
                'NE': (680, 693),
                'NV': (889, 899),
                'NH': ('030', '038'),
                'NJ': ('070', '089'),
                'NM': (870, 884),
                'NY': (100, 149),
                'NC': (270, 289),
                'ND': (580, 588),
                'OH': (430, 459),
                'OK': (730, 749),
                'OR': (970, 979),
                'PA': (150, 196),
                'RI': ('028', '029'),
                'SC': (290, 299),
                'SD': (570, 577),
                'TN': (370, 385),
                'TX': (750, 799),
                'UT': (840, 847),
                'VT': ('050', '059'),
                'VA': (220, 246),
                'WA': (980, 994),
                'WV': (247, 268),
                'WI': (530, 549),
                'WY': (820, 831)
            }
            
            for state, ranges in state_ranges.items():
                # If ranges is a list of tuples (for states with multiple ranges)
                if isinstance(ranges, list):
                    for lower, upper in ranges:
                        # Convert to integers for comparison if they're strings
                        lower_val = int(lower) if isinstance(lower, str) else lower
                        upper_val = int(upper) if isinstance(upper, str) else upper
                        
                        if lower_val <= zip_prefix <= upper_val:
                            logger.info(f"Found state {state} for zip code {zip_code} using prefix range mapping")
                            return state
                # If it's a single tuple range
                else:
                    lower, upper = ranges
                    # Convert to integers for comparison if they're strings
                    lower_val = int(lower) if isinstance(lower, str) else lower
                    upper_val = int(upper) if isinstance(upper, str) else upper
                    
                    if lower_val <= zip_prefix <= upper_val:
                        logger.info(f"Found state {state} for zip code {zip_code} using prefix range mapping")
                        return state
            
            logger.warning(f"Could not determine state for zip code: {zip_code}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting state from zip code {zip_code}: {str(e)}")
            return None
        
    def _is_abortion_related(self, message):
        """Check if a message is related to abortion access or policy"""
        if not message:
            return False
        
        message_lower = message.lower()
        
        # Define patterns that indicate abortion-related context
        abortion_terms = [
            'abortion', 'terminate pregnancy', 'pregnancy termination', 
            'end pregnancy', 'end a pregnancy', 'end my pregnancy',
            'abortion pill', 'abortion clinic', 'abortion provider',
            'roe v wade', 'legal abortion', 'abortion access', 'abortion restriction',
            'abortion law', 'abortion policy', 'abortion service', 'abortion right',
            'abortion ban', 'abortion legal', 'abortion illegal'
        ]
        
        # Check for phrases asking about abortion access
        abortion_access_patterns = [
            'can i get an abortion', 'i need an abortion', 'where can i get an abortion', 
            'how to get an abortion', 'abortion near me', 'abortion clinic near',
            'abortion provider near', 'find abortion', 'need abortion', 'want abortion',
            'where to get abortion', 'how to access abortion', 'looking for abortion',
            'is abortion legal', 'is abortion available', 'is abortion allowed',
            'abortion options', 'abortion services', 'getting an abortion',
            'seeking abortion', 'access to abortion'
        ]
        
        # Check for generic terms in message
        if any(term in message_lower for term in abortion_terms):
            return True
        
        # Check for access patterns
        if any(pattern in message_lower for pattern in abortion_access_patterns):
            return True
        
        # Check for state policy questions
        if 'abortion' in message_lower and any(word in message_lower for word in ['state', 'legal', 'allowed', 'law', 'available']):
            return True
        
        return False
        
    def _has_recent_abortion_context(self):
        """Check if there's abortion-related context in recent messages"""
        if not self.conversation_history:
            return False
        
        # Check last 3 messages for abortion context
        for entry in reversed(self.conversation_history[-3:]):
            if entry.get('sender') == 'user':
                if self._is_abortion_related(entry.get('message', '')):
                    logger.info("Found abortion context in recent history")
                    return True
        
        return False

    def _get_state_policy_details(self, state_code):
        """Get policy details for a specific state"""
        if not state_code:
            return None
            
        # Check cache first to reduce API costs
        if state_code in self.policy_cache:
            logger.info(f"Using cached policy details for {state_code}")
            return self.policy_cache[state_code]
            
        try:
            # Get policy details from API
            policy_details = self.policy_api.get_state_policy(state_code)
            
            # Cache the result for future use
            self.policy_cache[state_code] = policy_details
            
            return policy_details
        except Exception as e:
            logger.error(f"Error getting policy details for {state_code}: {str(e)}")
            return None
        
    def _state_has_restrictions(self, policy_details):
        """Determine if a state has restrictive abortion laws based on policy details"""
        if not policy_details:
            return False
            
        # Check for restrictive policies
        restrictions = [
            policy_details.get('gestational_limit', 24) < 12,  # Restrictive if less than 12 weeks
            policy_details.get('waiting_period', 0) > 0,       # Has mandatory waiting period
            policy_details.get('parental_consent', False),     # Requires parental consent for minors
            policy_details.get('banned', False),               # Abortion is fully banned
            policy_details.get('restricted', False)            # Has other restrictions
        ]
        
        return any(restrictions)
        
    def _get_nearby_clinics(self, zip_code):
        """Get information about nearby abortion clinics based on zip code"""
        # Hardcoded clinic data for demo/testing purposes
        clinics_by_state = {
            'NY': [
                {
                    'name': 'Planned Parenthood - Manhattan Health Center',
                    'address': '26 Bleecker St, New York, NY 10012',
                    'phone': '(212) 965-7000',
                    'website': 'https://www.plannedparenthood.org/health-center/new-york/new-york/10012/manhattan-health-center-3325-91110',
                    'lat': 40.725,
                    'lng': -73.994,
                    'services': ['Medical abortion', 'Surgical abortion', 'Birth control']
                },
                {
                    'name': 'Planned Parenthood - Bronx Center',
                    'address': '349 E 149th St, Bronx, NY 10451',
                    'phone': '(212) 965-7000',
                    'website': 'https://www.plannedparenthood.org/health-center/new-york/bronx/10451/bronx-center-3325-91110',
                    'lat': 40.817,
                    'lng': -73.920,
                    'services': ['Medical abortion', 'Surgical abortion', 'Birth control']
                }
            ],
            'CA': [
                {
                    'name': 'Planned Parenthood - Los Angeles',
                    'address': '1016 N Vermont Ave, Los Angeles, CA 90029',
                    'phone': '(800) 576-5544',
                    'website': 'https://www.plannedparenthood.org/health-center/california/los-angeles/90029/los-angeles-health-center-2454-90070',
                    'lat': 34.090,
                    'lng': -118.291,
                    'services': ['Medical abortion', 'Surgical abortion', 'Birth control']
                }
            ],
            'TX': [
                {
                    'name': 'Whole Woman\'s Health of Austin',
                    'address': '4100 Duval Rd, Austin, TX 78759',
                    'phone': '(512) 284-2224',
                    'website': 'https://www.wholewomanshealth.com/clinic/texas-austin-abortion-clinic/',
                    'lat': 30.415,
                    'lng': -97.732,
                    'services': ['Counseling', 'Referrals to out-of-state providers']
                }
            ],
            'GA': [
                {
                    'name': 'Feminist Women\'s Health Center',
                    'address': '1924 Cliff Valley Way NE, Atlanta, GA 30329',
                    'phone': '(404) 728-7900',
                    'website': 'https://www.feministcenter.org/',
                    'lat': 33.817,
                    'lng': -84.322,
                    'services': ['Birth control', 'Pregnancy testing', 'Counseling']
                }
            ]
        }
        
        # Default clinic info for states not in our hardcoded list
        default_clinics = [
            {
                'name': 'Planned Parenthood - Find a Health Center',
                'address': 'Multiple locations available',
                'phone': '1-800-230-PLAN',
                'website': 'https://www.plannedparenthood.org/health-center',
                'services': ['Call for services information']
            }
        ]
        
        # Get state from zip code
        state_code = self._get_state_from_zip_code(zip_code)
        if not state_code:
            return default_clinics
            
        # Return clinics for the state or default if none found
        return clinics_by_state.get(state_code, default_clinics)
        
    def _get_out_of_state_options(self, state_code):
        """Get nearby states with less restrictive abortion laws"""
        if not state_code:
            return None
            
        # Make sure state_code is uppercase
        state_code = state_code.upper()
        
        # Map of restrictive states to their nearby states with better access
        # Updated to be comprehensive for all states
        nearby_states_map = {
            'AL': ['FL', 'GA', 'TN'],  # Alabama
            'AK': ['WA', 'OR'],  # Alaska (not bordering, but closest)
            'AZ': ['CA', 'NV', 'NM'],  # Arizona
            'AR': ['IL', 'MO', 'TN'],  # Arkansas
            'CA': ['NV', 'OR', 'WA'],  # California 
            'CO': ['NM', 'WY', 'NE'],  # Colorado
            'CT': ['NY', 'MA', 'RI'],  # Connecticut
            'DE': ['NJ', 'MD', 'PA'],  # Delaware
            'FL': ['GA'],  # Florida
            'GA': ['FL', 'NC', 'SC'],  # Georgia
            'HI': ['CA', 'WA'],  # Hawaii (not bordering, but closest)
            'ID': ['WA', 'OR', 'MT'],  # Idaho
            'IL': ['WI', 'IN', 'KY'],  # Illinois
            'IN': ['IL', 'OH', 'KY'],  # Indiana
            'IA': ['IL', 'MN', 'WI'],  # Iowa
            'KS': ['CO', 'NE', 'MO'],  # Kansas
            'KY': ['IL', 'IN', 'OH', 'VA'],  # Kentucky
            'LA': ['TX', 'MS', 'FL'],  # Louisiana
            'ME': ['NH', 'MA'],  # Maine
            'MD': ['DE', 'PA', 'VA'],  # Maryland
            'MA': ['NY', 'VT', 'NH', 'CT', 'RI'],  # Massachusetts
            'MI': ['IL', 'WI', 'MN'],  # Michigan
            'MN': ['WI', 'IA', 'IL'],  # Minnesota
            'MS': ['AL', 'TN', 'IL'],  # Mississippi
            'MO': ['IL', 'KS', 'NE'],  # Missouri
            'MT': ['ID', 'WA', 'OR'],  # Montana
            'NE': ['CO', 'MO', 'IA'],  # Nebraska
            'NV': ['CA', 'OR'],  # Nevada
            'NH': ['VT', 'MA'],  # New Hampshire
            'NJ': ['NY', 'DE', 'PA'],  # New Jersey
            'NM': ['CO', 'AZ'],  # New Mexico
            'NY': ['NJ', 'PA', 'MA', 'CT', 'VT'],  # New York
            'NC': ['VA', 'SC'],  # North Carolina
            'ND': ['MN', 'MT'],  # North Dakota
            'OH': ['PA', 'MI', 'IL'],  # Ohio
            'OK': ['KS', 'CO', 'NM'],  # Oklahoma
            'OR': ['WA', 'CA', 'NV'],  # Oregon
            'PA': ['NY', 'NJ', 'DE', 'MD'],  # Pennsylvania
            'RI': ['MA', 'CT'],  # Rhode Island
            'SC': ['NC', 'GA'],  # South Carolina
            'SD': ['MN', 'MT', 'WY'],  # South Dakota
            'TN': ['VA', 'NC', 'IL'],  # Tennessee
            'TX': ['NM', 'CO'],  # Texas
            'UT': ['CO', 'NV', 'NM'],  # Utah
            'VT': ['NY', 'NH', 'MA'],  # Vermont
            'VA': ['MD', 'NC', 'PA'],  # Virginia
            'WA': ['OR', 'CA'],  # Washington
            'WV': ['PA', 'MD', 'VA', 'OH'],  # West Virginia
            'WI': ['IL', 'MN', 'MI'],  # Wisconsin
            'WY': ['CO', 'MT']  # Wyoming
        }
        
        # For DC, special case as it's not a state
        if state_code == 'DC':
            return 'Virginia, Maryland, and New York'
            
        # States that have abortion bans or severe restrictions
        restrictive_states = [
            'AL', 'AR', 'ID', 'KY', 'LA', 'MS', 'MO', 'ND', 'OK', 
            'SD', 'TN', 'TX', 'UT', 'WV', 'WI'
        ]
        
        # States with good abortion access
        supportive_states = [
            'CA', 'CO', 'CT', 'DE', 'HI', 'IL', 'MA', 'MD', 
            'ME', 'MN', 'NJ', 'NM', 'NY', 'OR', 'RI', 'VT', 'WA'
        ]
        
        # If we're already in a supportive state, return None
        if state_code in supportive_states:
            return None
            
        # If the state is in our nearby states map
        if state_code in nearby_states_map:
            nearby = nearby_states_map[state_code]
            
            # Filter to only include supportive states
            supportive_nearby = [state for state in nearby if state in supportive_states]
            
            # If we found supportive nearby states
            if supportive_nearby:
                # Convert to state names
                state_names = [self.policy_api.STATE_NAMES.get(s, s) for s in supportive_nearby]
                if len(state_names) == 1:
                    return state_names[0]
                elif len(state_names) == 2:
                    return f"{state_names[0]} and {state_names[1]}"
                else:
                    return f"{', '.join(state_names[:-1])}, and {state_names[-1]}"
                    
            # If no supportive nearby states, return all nearby
            state_names = [self.policy_api.STATE_NAMES.get(s, s) for s in nearby]
            if len(state_names) == 1:
                return state_names[0]
            elif len(state_names) == 2:
                return f"{state_names[0]} and {state_names[1]}"
            else:
                return f"{', '.join(state_names[:-1])}, and {state_names[-1]}"
        
        # Default to closest supportive states if not in map
        default_options = ", ".join([self.policy_api.STATE_NAMES[code] for code in ['CA', 'NY', 'IL']])
        return default_options

    def _format_abortion_policy_response(self, state_code, policy_details, zip_code=None, clinics_info=None, has_restrictions=None, out_of_state_options=None):
        """Format the abortion policy response in a clear, structured way."""
        if not policy_details:
            return "I apologize, but I don't have current policy information for that state. Please consult official sources or healthcare providers for accurate information."

        state_name = us.states.lookup(state_code).name if state_code else "your state"
        response_parts = []

        # Opening statement
        response_parts.append(f"Here's what I know about abortion access in {state_name}:")

        # Legal Status
        legal_status = policy_details.get("legal_status", "").strip()
        if legal_status:
            response_parts.append(f"\n📋 *Current Legal Status:*\n{legal_status}")

        # Gestational Limits
        gestational_limit = policy_details.get("gestational_limit", "").strip()
        if gestational_limit:
            response_parts.append(f"\n⏱️ *Time Limits:*\n{gestational_limit}")

        # Requirements and Restrictions
        requirements = policy_details.get("requirements", "").strip()
        if requirements:
            response_parts.append(f"\n📝 *Requirements:*\n{requirements}")

        # Insurance Coverage
        insurance = policy_details.get("insurance_coverage", "").strip()
        if insurance:
            response_parts.append(f"\n💳 *Insurance Coverage:*\n{insurance}")

        # Clinic Information
        if clinics_info and clinics_info.get("clinics"):
            clinics = clinics_info["clinics"][:3]  # Show top 3 clinics
            response_parts.append("\n🏥 *Nearby Healthcare Providers:*")
            for clinic in clinics:
                clinic_info = (f"\n• {clinic['name']}\n"
                             f"  📍 {clinic['address']}\n"
                             f"  📞 {clinic['phone']}")
                response_parts.append(clinic_info)

        # Out of State Options
        if has_restrictions and out_of_state_options:
            response_parts.append("\n🚗 *Nearest Out-of-State Options:*")
            for state, distance in out_of_state_options[:2]:  # Show top 2 options
                response_parts.append(f"\n• {state} (approximately {distance} miles away)")

        # Resources and Support
        response_parts.append("\n📱 *Additional Resources:*")
        response_parts.append("• National Abortion Federation Hotline: 1-800-772-9100")
        response_parts.append("• Planned Parenthood: 1-800-230-PLAN")

        # Disclaimer
        response_parts.append("\n⚠️ *Please Note:* This information is subject to change. Always verify with healthcare providers or legal professionals for the most current information.")

        return "\n".join(response_parts)

    def _get_predefined_response(self, message_lower):
        """Get predefined response for common messages without API calls or safety checks"""
        # Common greetings and simple responses
        predefined_responses = {
            "hello": "Hello! I'm Abby, here to provide information about reproductive healthcare. How can I help you today?",
            "hi": "Hi there! I'm Abby, and I'm here to answer your questions about reproductive health. What would you like to know?",
            "hey": "Hey! I'm Abby. I can provide you with information about reproductive health topics. What questions do you have?",
            "thanks": "You're welcome! Is there anything else I can help you with?",
            "thank you": "You're welcome! I'm here if you need more information.",
            "goodbye": "Take care! If you have more questions in the future, feel free to reach out.",
            "bye": "Goodbye! I'm here if you need more information later."
        }
        
        # Exact matches for simple queries
        if message_lower.strip() in predefined_responses:
            response = predefined_responses[message_lower.strip()]
            message_id = str(uuid.uuid4())
            formatted_response = {
                "text": response,
                "citations": [],
                "citation_objects": [],
                "citations_html": "",
                "message_id": message_id
            }
            
            # Add to history without evaluation
            message_entry = {
                'session_id': self.current_session_id,
                'message_id': message_id,
                'sender': 'bot',
                'message': response,
                'timestamp': datetime.now().isoformat(),
                'type': 'message'
            }
            self.conversation_history.append(message_entry)
            self.context_manager.add_message(message_entry)
            
            return formatted_response
            
        # How are you patterns
        how_are_you_patterns = ["how are you", "how's it going", "how are things", "how do you feel"]
        if any(pattern in message_lower for pattern in how_are_you_patterns) and len(message_lower.split()) <= 5:
            response = "I'm doing well, thanks for asking! I'm here to help with any reproductive health questions you might have. What can I assist you with today?"
            message_id = str(uuid.uuid4())
            formatted_response = {
                "text": response,
                "citations": [],
                "citation_objects": [],
                "citations_html": "",
                "message_id": message_id
            }
            
            # Add to history without evaluation
            message_entry = {
                'session_id': self.current_session_id,
                'message_id': message_id,
                'sender': 'bot',
                'message': response,
                'timestamp': datetime.now().isoformat(),
                'type': 'message'
            }
            self.conversation_history.append(message_entry)
            self.context_manager.add_message(message_entry)
            
            return formatted_response
        
        # Generic abortion access query without state
        abortion_access_patterns = ["can i get an abortion", "abortion in my state", "get an abortion", 
                                 "abortion access", "abortion near me", "abortion options", 
                                 "how to get an abortion", "where can i get an abortion", "my state"]
        
        # Check for generic abortion access query without state
        if any(pattern in message_lower for pattern in abortion_access_patterns):
            # Check if state is mentioned
            has_state_mention = False
            for state_name in self.policy_api.STATE_NAMES.values():
                if state_name.lower() in message_lower:
                    has_state_mention = True
                    break
                    
            for state_code in self.policy_api.STATE_NAMES.keys():
                # Skip 'IN' (Indiana) when checking for state codes, as it's often used as a preposition
                if state_code.lower() == 'in' and ("in my state" in message_lower or "my state" in message_lower):
                    continue
                
                if state_code.lower() in message_lower:
                    has_state_mention = True
                    break
            
            # Only use predefined response if no state is mentioned
            if not has_state_mention:
                standard_abortion_response = (
                    "I understand that you're looking for information about abortion access, and it's important to have accurate details. "
                    "Abortion laws and availability vary widely across the U.S.—some states have restrictions, while others offer broader access. "
                    "Knowing the specific regulations in your state can help you understand your options.\n\n"
                    "If you're comfortable sharing your state, I can provide more detailed information on local laws and available services. "
                    "States may have different gestational limits, waiting periods, and requirements for minors. "
                    "Some also have specific clinics or healthcare providers that offer abortion services.\n\n"
                    "If you're looking for nearby clinics, I can help with that too—just let me know your city or state, and I'll find the closest options for you. "
                )
                
                message_id = str(uuid.uuid4())
                formatted_response = {
                    "text": standard_abortion_response,
                    "citations": [],
                    "citation_objects": [],
                    "citations_html": "",
                    "message_id": message_id
                }
                
                # Add to history directly without evaluation
                message_entry = {
                    'session_id': self.current_session_id,
                    'message_id': message_id,
                    'sender': 'bot',
                    'message': standard_abortion_response,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'message'
                }
                self.conversation_history.append(message_entry)
                if hasattr(self, 'context_manager'):
                    self.context_manager.add_message(message_entry)
                
                logger.info("Using standardized abortion access response for general query")
                return formatted_response
                
        return None

    def _preprocess_message(self, message):
        """
        Preprocesses the user message before main processing
        
        Args:
            message (str): User message to preprocess
            
        Returns:
            dict: Preprocessed message with metadata
        """
        # Check if message is in English
        language = detect_language(message)
        
        if language != 'en':
            logger.info(f"Non-English language detected: {language}")
            message_id = str(uuid.uuid4())
            non_english_response = (
                "I'm sorry, but I currently only support English. "
                "Please provide your question in English so I can assist you properly."
            )
            
            # Add to conversation history
            self.add_to_history('user', message)
            
            response = {
                'text': non_english_response,
                'citations': [],
                'citation_objects': [],
                'message_id': message_id
            }
            
            # Add bot response to history
            self.add_to_history('bot', non_english_response, message_id=message_id)
            
            result = {
                "original": message,
                "text": message,
                "language": language,
                "preprocessed_response": response
            }
            
            return result
        
        # Check for PII
        pii_detector = PIIDetector()
        has_pii = pii_detector.has_pii(message)
        sanitized_message = message
        pii_warning = None
        
        if has_pii:
            sanitized_message, pii_warning = pii_detector.detect_and_sanitize(message)
        
        result = {
            "original": message,
            "text": sanitized_message if has_pii else message,
            "has_pii": has_pii,
            "language": "en"
        }
        
        # If PII detected, return a warning response
        if has_pii:
            logger.warning("PII detected in user message - preprocessing complete")
            if not pii_warning:
                pii_warning = (
                    "• I noticed you shared personal information in your message.\n"
                    "• For privacy and security reasons, please avoid sharing personal details.\n"
                    "• Is there something specific about reproductive health I can help you with?"
                )
            
            # Add to conversation history with PII redacted
            self.add_to_history('user', "[PII DETECTED - MESSAGE REDACTED]")
            
            message_id = str(uuid.uuid4())
            response = {
                'text': pii_warning,
                'citations': [],
                'citation_objects': [],
                'message_id': message_id
            }
            
            # Add bot response to history
            self.add_to_history('bot', pii_warning, message_id=message_id)
            
            result["preprocessed_response"] = response
            
        return result

    def add_to_history(self, sender, message, message_id=None, evaluate=False):
        """
        Add a message to the conversation history
        
        Args:
            sender (str): The sender of the message ('user' or 'bot')
            message (str): The message content
            message_id (str, optional): Unique ID for the message
            evaluate (bool): Whether to evaluate this message
            
        Returns:
            str: The message_id
        """
        # Generate ID if not provided
        if not message_id:
            message_id = str(uuid.uuid4())
            
        # Check for PII in user messages and sanitize if needed
        sanitized_message = message
        if sender == 'user':
            # Use our updated PII detector that preserves zip codes and state information
            has_pii = self.pii_detector.has_pii(message)
            
            if has_pii:
                # Get types of PII detected
                pii_types = self.pii_detector.detect_pii_types(message)
                logger.warning(f"PII types detected in user message: {pii_types}")
                
                # Sanitize the message while preserving zip codes and state information
                sanitized_message, redacted_items = self.pii_detector.redact_pii(message)
                logger.info(f"Sanitized user message with {len(redacted_items)} PII items redacted")
                logger.debug(f"Original: '{message}' -> Sanitized: '{sanitized_message}'")
                
                # Track that PII was detected for metrics
                self.metrics_tracker.track_pii_detection(len(redacted_items), pii_types)
            
        # Create message entry with timestamp
        timestamp = datetime.now().timestamp()
        
        message_entry = {
            'session_id': self.current_session_id,
            'message_id': message_id,
            'sender': sender,
            'message': sanitized_message,  # Store sanitized version
            'original_message': message if sanitized_message != message else None,  # Store original only if different
            'timestamp': timestamp,
            'type': 'message'
        }
        
        # For bot messages, periodically evaluate to reduce cost but collect data
        if sender == 'bot' and evaluate and random.random() < 0.25:  # 25% chance
            evaluation_data = self._evaluate_bot_message(sanitized_message)
            if evaluation_data:
                message_entry['evaluation'] = evaluation_data
        
        # Add to conversation history
        self.conversation_history.append(message_entry)
        
        # Also add to context manager for tracking recent context
        if hasattr(self, 'context_manager'):
            self.context_manager.add_message(message_entry)
            
        # Save conversation logs after each message to ensure nothing is lost
        self._save_conversation_logs()
            
        return message_id

    def get_history(self):
        """
        Get the conversation history
        
        If _session_ended is True, return an empty list to indicate 
        the UI should start a fresh conversation display.

        Returns:
            list: List of conversation messages or empty list if session ended
        """
        if self._session_ended:
            logger.info("Session marked as ended, returning empty history for UI")
            return []
            
        # Return the context from ContextManager instead of full history
        return self.context_manager.get_context()

    def clear_history(self):
        """
        Archive the conversation history and prepare for a new session
        
        This method is called when the user ends their session.
        Instead of deleting the history, we archive it for feedback and logs,
        but mark the session as complete for the UI.

        Returns:
            bool: True if operation was successful
        """
        try:
            # Archive the conversation for analytics but don't delete it
            # Just log that we're ending the session for this user
            if self.conversation_history:
                logger.info(f"Session ended with {len(self.conversation_history)} messages")
                
                # Save the conversation logs before marking session as ended
                self._save_conversation_logs()
                
                # Clear both conversation history and context
                self.conversation_history = []
                self.context_manager.clear_context()
                
                # Preserve the history for analytics but set a flag so the UI knows to start fresh
                # We don't actually clear the history, as that would lose context
                self._session_ended = True
                
                # In a future enhancement, we could store the complete history in a database
                # with a session ID for better analytics
            
            # Return success - UI will handle clearing the display
            return True
        except Exception as e:
            logger.error(f"Error handling session end: {str(e)}")
            return False

    def detect_location_context(self, message: str) -> Optional[str]:
        """
        Detect if a US location is mentioned in the message.
        Also identifies international locations for proper handling.

        Args:
            message (str): The user message to analyze

        Returns:
            Optional[str]: The detected location (US state or international) 
                          or None if no location is found
        """
        message_lower = message.lower()

        # Special case for "Indiana" since it can be confused with "India"
        if "indiana" in message_lower:
            logger.info(f"Found direct state mention in message: Indiana")
            return "indiana"

        # Check for direct mentions of US states
        for code, state in self.policy_api.STATE_NAMES.items():
            if state.lower() in message_lower:
                logger.info(f"Found direct state mention in message: {state}")
                return state.lower()

        # First check for state abbreviations (two-letter codes)
        # We need to be careful with 'IN' since it's commonly used as a preposition
        words = message_lower.split()
        for word in words:
            word_cleaned = word.strip('.,?!;:()')
            if word_cleaned.upper() in self.policy_api.STATE_NAMES and word_cleaned.upper() != 'IN':
                state_code = word_cleaned.upper()
                state_name = self.policy_api.STATE_NAMES[state_code]
                logger.info(f"Found state abbreviation in message: {state_code} ({state_name})")
                return state_name.lower()
                
        # Special case for 'IN' (Indiana) - only count if it appears to be a state reference
        # rather than the preposition "in"
        if any(phrase in message_lower for phrase in ["in state", "state of in", "in abortion"]):
            logger.info(f"Found specific reference to Indiana")
            return "indiana"
            
        # Specifically check if preposition "in" is used with "my state" to avoid detecting Indiana
        if "in my state" in message_lower or "in your state" in message_lower:
            logger.info("Found 'in my state' phrase, NOT treating 'in' as Indiana")
            return None
            
        # Look for phrases that might indicate a state reference
        state_indicator_phrases = [
            "i live in", "i'm in", "i am in", "my state is", "in the state of",
            "laws in", "policies in", "i'm from", "i am from"
        ]
        
        for phrase in state_indicator_phrases:
            if phrase in message_lower:
                # Extract what comes after the phrase
                phrase_index = message_lower.find(phrase) + len(phrase)
                remainder = message_lower[phrase_index:].strip()
                first_word = remainder.split()[0] if remainder and ' ' in remainder else remainder
                
                # Check if this first word is a state abbreviation
                if first_word.upper() in self.policy_api.STATE_NAMES:
                    state_code = first_word.upper()
                    state_name = self.policy_api.STATE_NAMES[state_code]
                    logger.info(f"Found state abbreviation after location phrase: {state_code} ({state_name})")
                    return state_name.lower()
                
                # Check the first few words against state names
                for i in range(1, min(4, len(remainder.split()) + 1)):
                    potential_state = ' '.join(remainder.split()[:i])
                    for state_name in self.policy_api.STATE_NAMES.values():
                        if potential_state.lower() == state_name.lower():
                            logger.info(f"Found state name after location phrase: {state_name}")
                            return state_name.lower()
                
        # Check for zip codes - zip codes can indicate location without being PII
        import re
        zip_matches = re.findall(r'\b\d{5}(?:-\d{4})?\b', message)
        if zip_matches:
            logger.info(f"Found ZIP code in message: {zip_matches[0]}")
            # We don't directly return a location from a ZIP code,
            # but we could use a ZIP code database to look up the state
            # For now, just note that we found a ZIP code
            
        # Comprehensive list of non-US countries
        non_us_countries = {
            'afghanistan', 'albania', 'algeria', 'andorra', 'angola', 'antigua and barbuda', 'argentina', 
            'armenia', 'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain', 'bangladesh', 'barbados', 
            'belarus', 'belgium', 'belize', 'benin', 'bhutan', 'bolivia', 'bosnia and herzegovina', 
            'botswana', 'brazil', 'brunei', 'bulgaria', 'burkina faso', 'burundi', 'cabo verde', 'cambodia', 
            'cameroon', 'canada', 'central african republic', 'chad', 'chile', 'china', 'colombia', 'comoros', 
            'congo', 'costa rica', 'croatia', 'cuba', 'cyprus', 'czech republic', 'denmark', 'djibouti', 
            'dominica', 'dominican republic', 'ecuador', 'egypt', 'el salvador', 'equatorial guinea', 
            'eritrea', 'estonia', 'eswatini', 'ethiopia', 'fiji', 'finland', 'france', 'gabon', 'gambia', 
            'georgia', 'germany', 'ghana', 'greece', 'grenada', 'guatemala', 'guinea', 'guinea-bissau', 
            'guyana', 'haiti', 'honduras', 'hungary', 'iceland', 'india', 'indonesia', 'iran', 'iraq', 
            'ireland', 'israel', 'italy', 'jamaica', 'japan', 'jordan', 'kazakhstan', 'kenya', 'kiribati', 
            'korea', 'north korea', 'south korea', 'kosovo', 'kuwait', 'kyrgyzstan', 'laos', 'latvia', 
            'lebanon', 'lesotho', 'liberia', 'libya', 'liechtenstein', 'lithuania', 'luxembourg', 
            'madagascar', 'malawi', 'malaysia', 'maldives', 'mali', 'malta', 'marshall islands', 
            'mauritania', 'mauritius', 'mexico', 'micronesia', 'moldova', 'monaco', 'mongolia', 
            'montenegro', 'morocco', 'mozambique', 'myanmar', 'namibia', 'nauru', 'nepal', 'netherlands', 
            'new zealand', 'nicaragua', 'niger', 'nigeria', 'north macedonia', 'norway', 'oman', 'pakistan', 
            'palau', 'palestine', 'panama', 'papua new guinea', 'paraguay', 'peru', 'philippines', 'poland', 
            'portugal', 'qatar', 'romania', 'russia', 'rwanda', 'saint kitts and nevis', 'saint lucia', 
            'saint vincent and the grenadines', 'samoa', 'san marino', 'sao tome and principe', 
            'saudi arabia', 'senegal', 'serbia', 'seychelles', 'sierra leone', 'singapore', 'slovakia', 
            'slovenia', 'solomon islands', 'somalia', 'south africa', 'south sudan', 'spain', 'sri lanka', 
            'sudan', 'suriname', 'sweden', 'switzerland', 'syria', 'taiwan', 'tajikistan', 'tanzania', 
            'thailand', 'timor-leste', 'togo', 'tonga', 'trinidad and tobago', 'tunisia', 'turkey', 
            'turkmenistan', 'tuvalu', 'uganda', 'ukraine', 'united arab emirates', 'united kingdom', 'uk',
            'uruguay', 'uzbekistan', 'vanuatu', 'vatican city', 'venezuela', 'vietnam', 'yemen', 'zambia', 
            'zimbabwe'
        }

        # Check for international countries 
        for country in non_us_countries:
            # Skip 'india' if 'indiana' is in the text to avoid confusion
            if country == 'india' and 'indiana' in message_lower:
                continue
                
            if country in message_lower:
                logger.info(f"Found international location mention in message: {country}")
                return country
                
        return None

    def clean_message(self, message):
        """Clean and preprocess a message"""
        if not message:
            return ""
        
        # Convert to string if not already
        message = str(message)
        
        # Remove extra whitespace
        message = " ".join(message.split())
        
        # Remove special characters but keep basic punctuation
        message = re.sub(r'[^\w\s.,!?-]', '', message)
        
        return message.strip()

    def is_state_only_message(self, content):
        """Check if the message is just a state name or abbreviation"""
        if not content:
            return False
        
        content = content.strip().lower()
        
        # Check if content matches a state name or abbreviation
        state_names_lower = [name.lower() for name in self.policy_api.STATE_NAMES.values()]
        state_abbrevs_lower = [abbr.lower() for abbr in self.policy_api.STATE_NAMES.keys()]
        
        # Special case for multi-word state names like "New York"
        is_state_name = content in state_names_lower or content in state_abbrevs_lower
        
        # For multi-word state names, do additional check
        if not is_state_name:
            for state_name in state_names_lower:
                if content in state_name or state_name in content:
                    # Check if it's a close match (e.g., "new york" matches "New York")
                    if content.split() and state_name.split():
                        # Check first word matches
                        if content.split()[0] == state_name.split()[0]:
                            is_state_name = True
                            logger.info(f"Matched partial state name: {content} -> {state_name}")
                            break
        
        if is_state_name:
            logger.info(f"Detected state-only message: {content}")
            
            # Check if there's abortion context in recent history
            for entry in reversed(self.conversation_history[-3:]):  # Check last 3 messages
                if entry.get('sender') == 'user':
                    if self._is_abortion_related(entry.get('message', '')):
                        logger.info(f"State-only message with abortion context in history: {content}")
                        return 'abortion_context'
            
            # Also check bot messages for abortion context (like asking for state)
            for entry in reversed(self.conversation_history[-3:]):
                if entry.get('sender') == 'bot':
                    bot_msg = entry.get('message', '').lower()
                    if ('abortion' in bot_msg or 'policy' in bot_msg) and ('which state' in bot_msg or 'what state' in bot_msg or 'your state' in bot_msg):
                        logger.info(f"State-only message with bot abortion prompt in history: {content}")
                        return 'abortion_context'
                    
            return 'state_only'
        
        return False

    def handle_message(self, content):
        """Process user message and generate appropriate response"""
        start_time = time.time()

        # Record original query for metrics
        original_query = content

        # Clean and preprocess the message
        content = self.clean_message(content)

        # Check if the message is a state name only
        is_state_only = self.is_state_only_message(content)

        # Check if PII should be detected before abort policy questions
        if not is_state_only:
            # Check if message contains PII and sanitize if needed
            sanitized_content, warning = self.pii_detector.detect_and_sanitize(content)
            if warning:
                logger.info("PII detected and sanitized in message")
                content = sanitized_content

        # If the message is just a state name, convert it to a policy question
        if is_state_only:
            logger.info(f"Detected state-only message: {content}")
            # Extract the state code
            state_code = None
            content_lower = content.lower()
            
            # First check for exact matches with state names
            for code, name in self.policy_api.STATE_NAMES.items():
                if content_lower == name.lower():
                    state_code = code
                    break
            
            # If no exact match, check for partial matches (like "new york" matching "New York")
            if not state_code:
                for code, name in self.policy_api.STATE_NAMES.items():
                    name_lower = name.lower()
                    # Check if content is a partial match
                    if content_lower in name_lower or name_lower in content_lower:
                        # For multi-word states, verify first word matches
                        if content_lower.split() and name_lower.split():
                            if content_lower.split()[0] == name_lower.split()[0]:
                                state_code = code
                                logger.info(f"Partial state name match: {content_lower} -> {name_lower}")
                                break
            
            # Check direct match with state abbreviation if we still don't have a match
            if not state_code and content_lower.upper() in self.policy_api.STATE_NAMES:
                state_code = content_lower.upper()
            
            if state_code:
                logger.info(f"Converting state-only message to policy question for state code: {state_code}")
                
                # If we have abortion context, get policy information
                if is_state_only == 'abortion_context':
                    logger.info(f"Processing abortion policy request for state: {state_code}")
                    policy_response = self._get_policy_response_for_state(state_code, str(uuid.uuid4()))
                    return policy_response
                else:
                    # Create a general response asking how to help
                    response = {
                        'text': f"I see you're asking about {self.policy_api.STATE_NAMES.get(state_code, content)}. How can I help you with information about this state? I can provide details about reproductive healthcare services, policies, or other health-related topics.",
                        'citations': [],
                        'citation_objects': [],
                        'message_id': str(uuid.uuid4())
                    }
                    return response

        # Process the message normally
        response = self.process_message(content)

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Total message processing time: {processing_time:.2f} seconds")

        return response

    def _format_response(self, response: str, empathy_level: str = "high", is_policy_response: bool = False) -> str:
        """Format the response with appropriate empathy and natural paragraphs"""
        
        # Add empathetic prefix based on level
        empathy_prefixes = {
            "high": [
                "I want you to know I'm here for you during this challenging time.",
                "I understand this is a deeply personal decision, and I'm here to support you.",
                "I truly care about helping you through this difficult situation.",
                "I'm listening and I want to support you in any way I can right now.",
                "Your feelings matter, and I'm here to provide the information and support you need.",
                "I completely understand why this would feel overwhelming, and I'm here to help.",
            ],
            "medium": [
                "I hear your concerns and want to provide support.",
                "Thank you for trusting me with your questions.",
                "I'm here to help you navigate this situation with care.",
                "I understand this topic can bring up many emotions.",
                "I appreciate you reaching out, and I'm here to help.",
            ],
            "low": [
                "I'm here to provide helpful information with care.",
                "I'm happy to help with the information you need.",
                "I can provide supportive guidance on this topic.",
                "I'm here to offer information and support.",
            ]
        }
        
        import random
        prefix = random.choice(empathy_prefixes.get(empathy_level, empathy_prefixes["high"]))
        
        # Format the main response
        lines = response.split('\n')
        formatted_lines = []
        
        # Add the empathetic prefix
        formatted_lines.append(prefix)
        formatted_lines.append("")  # Add blank line after prefix
        
        # Format remaining lines into paragraphs
        current_paragraph = []
        for line in lines:
            line = line.strip()
            if line:
                # Skip empty lines and lines that are just bullet points
                if not line or line == '•':
                    continue
                    
                # Skip if line is the same as prefix
                if line == prefix:
                    continue
                    
                # Remove bullet points if present
                if line.startswith('•'):
                    line = line[1:].strip()
                
                current_paragraph.append(line)
            elif current_paragraph:
                # Join current paragraph and add to formatted lines
                formatted_lines.append(' '.join(current_paragraph))
                current_paragraph = []
                formatted_lines.append("")  # Add blank line between paragraphs
        
        # Add any remaining paragraph
        if current_paragraph:
            formatted_lines.append(' '.join(current_paragraph))
            formatted_lines.append("")
        
        # Add standard resources at the end if not already present
        if "resources" not in response.lower():
            formatted_lines.extend([
                "Helpful Resources:",
                "Planned Parenthood: 1-800-230-PLAN",
                "www.ineedana.com - Find local resources and clinics",
                "National Abortion Federation Hotline: 1-800-772-9100",
                "Remember, you're not alone in this journey, and support is available to help you at every step."
            ])
        
        return '\n'.join(formatted_lines)

    def _process_policy_question(self, question: str, state_code: str = None) -> str:
        """Process policy-related questions with appropriate formatting"""
        try:
            # Check cache first to reduce API calls
            cache_key = f"{state_code}_{question[:50]}"  # First 50 chars as key with state
            if cache_key in self.policy_cache:
                logger.info(f"Using cached policy response for {state_code}")
                return self.policy_cache[cache_key]
            
            response = self.policy_api.get_policy_response(question, state_code=state_code)
            
            # Cache the response
            self.policy_cache[cache_key] = response
            
            # If it's already formatted with bullets, return as is
            if '•' in response:
                return response
                
            # Otherwise, format it with bullets
            return self._format_response(response, empathy_level="medium", is_policy_response=True)
            
        except Exception as e:
            logger.error(f"Error processing policy question: {str(e)}")
            return self._format_response(
                "I apologize, but I'm having trouble accessing the policy information right now. "
                "Please try again later or contact Planned Parenthood directly for the most current information.",
                empathy_level="high",
                is_policy_response=True
            )
            
    def _check_simple_queries(self, message: str) -> dict:
        """
        Check for simple, common queries that can be answered without API calls
        
        Args:
            message (str): The user message (lowercase)
            
        Returns:
            dict: Response object if a simple query is matched, None otherwise
        """
        # Common pre-defined responses to avoid API costs
        common_responses = {
            "hello": "Hello! I'm Abby, here to provide information about reproductive healthcare. How can I help you today?",
            "hi": "Hi there! I'm Abby, and I'm here to answer your questions about reproductive health. What would you like to know?",
            "hey": "Hey! I'm Abby. I can provide you with information about reproductive health topics. What questions do you have?",
            "thanks": "You're welcome! Is there anything else I can help you with?",
            "thank you": "You're welcome! I'm here if you need more information.",
            "goodbye": "Take care! If you have more questions in the future, feel free to reach out.",
            "bye": "Goodbye! I'm here if you need more information later."
        }
        
        # Check for exact matches
        for key, response in common_responses.items():
            if message.strip() == key:
                # Create a direct response without any processing
                message_id = str(uuid.uuid4())
                # Skip the add_to_history call to avoid evaluation
                formatted_response = {
                    "text": response,
                    "citations": [],
                    "citation_objects": [],
                    "citations_html": "",
                    "message_id": message_id
                }
                # Manually add to history without evaluation
                message_entry = {
                    'session_id': self.current_session_id,
                    'message_id': message_id,
                    'sender': 'bot',
                    'message': response,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'message'
                }
                self.conversation_history.append(message_entry)
                if hasattr(self, 'context_manager'):
                    self.context_manager.add_message(message_entry)
                
                return formatted_response
                
        # Check for simple how are you questions
        how_are_you_patterns = ["how are you", "how's it going", "how are things", "how do you feel"]
        if any(pattern in message for pattern in how_are_you_patterns) and len(message.split()) <= 5:
            response = "I'm doing well, thanks for asking! I'm here to help with any reproductive health questions you might have. What can I assist you with today?"
            # Create a direct response without any processing
            message_id = str(uuid.uuid4())
            # Skip the add_to_history call to avoid evaluation
            formatted_response = {
                "text": response,
                "citations": [],
                "citation_objects": [],
                "citations_html": "",
                "message_id": message_id
            }
            # Manually add to history without evaluation
            message_entry = {
                'session_id': self.current_session_id,
                'message_id': message_id,
                'sender': 'bot',
                'message': response,
                'timestamp': datetime.now().isoformat(),
                'type': 'message'
            }
            self.conversation_history.append(message_entry)
            if hasattr(self, 'context_manager'):
                self.context_manager.add_message(message_entry)
            
            return formatted_response
            
        # Check for general abortion access questions without a specific state
        abortion_access_patterns = ["can i get an abortion", "abortion in my state", "get an abortion", 
                                 "abortion access", "abortion near me", "abortion options", 
                                 "how to get an abortion", "where can i get an abortion", "my state"]
        
        # Check if this is a general abortion question without state
        if any(pattern in message for pattern in abortion_access_patterns):
            # Check if a state is mentioned
            has_state_mention = False
            for state_name in self.policy_api.STATE_NAMES.values():
                if state_name.lower() in message:
                    has_state_mention = True
                    break
                    
            for state_code in self.policy_api.STATE_NAMES.keys():
                # Skip 'IN' (Indiana) when checking for state codes, as it's often used as a preposition
                if state_code.lower() == 'in' and ("in my state" in message or "my state" in message):
                    continue
                
                if state_code.lower() in message:
                    has_state_mention = True
                    break
            
            # Only use predefined response if no state is mentioned
            if not has_state_mention:
                standard_abortion_response = (
                    "I understand that you're looking for information about abortion access, and it's important to have accurate details. "
                    "Abortion laws and availability vary widely across the U.S.—some states have restrictions, while others offer broader access. "
                    "Knowing the specific regulations in your state can help you understand your options.\n\n"
                    "If you're comfortable sharing your state, I can provide more detailed information on local laws and available services. "
                    "States may have different gestational limits, waiting periods, and requirements for minors. "
                    "Some also have specific clinics or healthcare providers that offer abortion services.\n\n"
                    "If you're looking for nearby clinics, I can help with that too—just let me know your city or state, and I'll find the closest options for you. "
                )
                
                message_id = str(uuid.uuid4())
                formatted_response = {
                    "text": standard_abortion_response,
                    "citations": [],
                    "citation_objects": [],
                    "citations_html": "",
                    "message_id": message_id
                }
                
                # Add to history directly without evaluation
                message_entry = {
                    'session_id': self.current_session_id,
                    'message_id': message_id,
                    'sender': 'bot',
                    'message': standard_abortion_response,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'message'
                }
                self.conversation_history.append(message_entry)
                if hasattr(self, 'context_manager'):
                    self.context_manager.add_message(message_entry)
                
                logger.info("Using standardized abortion access response for general query")
                return formatted_response
            
        return None

    def _extract_location(self, message, context=[]):
        """
        Extract location information from a message or context
        
        Args:
            message (str): User message to analyze
            context (list): Conversation context
            
        Returns:
            str: Location code (state code) or None if not found
        """
        # First check for direct state mentions in the message
        state_context = self._check_for_state_names(message)
        if state_context:
            logger.info(f"Found direct state mention: {state_context}")
            return state_context
            
        # Check for context clues like "I live in X" or "I'm from X"
        location_phrases = [
            r"(?:I(?:'m| am) (?:from|in)) ([A-Za-z ]+)",
            r"(?:I live in) ([A-Za-z ]+)",
            r"(?:my state is) ([A-Za-z ]+)",
            r"(?:my state) ([A-Za-z ]+)",
            r"(?:state of) ([A-Za-z ]+)"
        ]
        
        for phrase in location_phrases:
            match = re.search(phrase, message, re.IGNORECASE)
            if match:
                potential_location = match.group(1).strip()
                # Check if this matches a state name
                for code, state in self.policy_api.STATE_NAMES.items():
                    if potential_location.lower() == state.lower() or potential_location.lower() == code.lower():
                        logger.info(f"Found location from context phrase: {potential_location}")
                        return state.lower()
        
        # Check recent context for state mentions
        if context:
            for entry in reversed(context):
                if entry.get('sender') == 'user':
                    state_in_context = self._check_for_state_names(entry.get('message', ''))
                    if state_in_context:
                        logger.info(f"Found state {state_in_context} in conversation context")
                        return state_in_context
        
        # No location found
        return None

    def _is_policy_lookup(self, message):
        """Determine if a message is asking about abortion policy information"""
        # Keywords related to policy questions (expanded for better recall)
        policy_keywords = [
            'policy', 'policies', 'legal', 'illegal', 'allow', 'allowed', 'law',
            'laws', 'restrict', 'restriction', 'restrictions', 'state', 'states',
            'access', 'can i get', 'where can i', 'option', 'options'
        ]
        
        # Check for abortion context
        is_abortion_context = self._is_abortion_related(message)
        
        # Check for policy keywords
        message_lower = message.lower()
        has_policy_keywords = any(keyword in message_lower for keyword in policy_keywords)
        
        # Specific check for emergency contraception or plan B questions
        emergency_contraception_keywords = [
            'emergency contraception', 'plan b', 'morning after', 'ella', 
            'morning-after pill', 'day after pill', 'contraceptive pill',
            'morning pill', 'after pill', 'emergency pill', 'morning after pill',
            'plan b pill', 'iud as emergency', 'copper iud emergency', 'emergency iud'
        ]
        is_emergency_contraception = any(keyword in message_lower for keyword in emergency_contraception_keywords)
        
        if is_emergency_contraception:
            logger.info("Emergency contraception question detected")
            # Return False as we want to handle these differently than policy questions
            return False
            
        return is_abortion_context and has_policy_keywords

    def _get_policy_response(self, message):
        """
        Get abortion policy information based on the message
        
        Args:
            message (str): User message to analyze
            
        Returns:
            dict: Response data with policy information or None if not applicable
        """
        message_lower = message.lower()
        message_id = str(uuid.uuid4())
        
        # Extract any state information from the message
        state_context = self._check_for_state_names(message)
        
        # Extract any zip code information
        zip_match = re.search(r'\b\d{5}\b', message)
        zip_code = zip_match.group(0) if zip_match else None
        
        # If we have a zip code but no state, try to get state from zip
        if zip_code and not state_context:
            state_code = self._get_state_from_zip_code(zip_code)
            if state_code:
                state_context = state_code
                logger.info(f"Converted zip code {zip_code} to state: {state_context}")
            else:
                logger.warning(f"Could not determine state from zip code: {zip_code}")
        
        # Check for "my state" or similar references
        my_state_patterns = ["my state", "where i live", "where i am", "in our state", "our state laws"]
        has_my_state_reference = any(pattern in message_lower for pattern in my_state_patterns)
        
        # If we have a "my state" reference but no state context, check conversation history
        if has_my_state_reference and not state_context:
            logger.info("User mentioned 'my state' - checking conversation history")
            conversation_history = self.get_history()
            
            # Look for state mentions in history
            for entry in reversed(conversation_history):
                if entry.get('sender') == 'user':
                    history_state = self._check_for_state_names(entry.get('message', ''))
                    if history_state:
                        state_context = history_state
                        logger.info(f"Found state in conversation history: {state_context}")
                        break
            
            # Also check for zip codes in history
            if not state_context:
                for entry in reversed(conversation_history):
                    if entry.get('sender') == 'user':
                        zip_match = re.search(r'\b\d{5}\b', entry.get('message', ''))
                        if zip_match:
                            history_zip = zip_match.group(0)
                            state_from_zip = self._get_state_from_zip_code(history_zip)
                            if state_from_zip:
                                state_context = state_from_zip
                                zip_code = history_zip
                                logger.info(f"Found zip code in history, converted to state: {state_context}")
                                break
        
        # If we have a state context, get state policy information
        if state_context:
            logger.info(f"Getting abortion policy information for state: {state_context}")
            try:
                # Get policy details from the API
                policy_details = self.policy_api.get_policy_data(state_context)
                
                # Check if the state has restrictions
                has_restrictions = self._state_has_restrictions(policy_details)
                
                # Get nearby clinics if we have a zip code
                clinics_info = None
                if zip_code:
                    clinics_info = self._get_nearby_clinics(zip_code)
                
                # Get out of state options if there are restrictions
                out_of_state_options = None
                if has_restrictions:
                    out_of_state_options = self._get_out_of_state_options(state_context)
                
                # Format the response with all the gathered information
                response = self._format_abortion_policy_response(
                    state_context,
                    policy_details,
                    zip_code=zip_code,
                    clinics_info=clinics_info,
                    has_restrictions=has_restrictions,
                    out_of_state_options=out_of_state_options
                )
                
                return {
                    'type': 'policy',
                    'message_id': message_id,
                    'response': response,
                    'state_context': state_context,
                    'zip_code': zip_code,
                    'has_restrictions': has_restrictions
                }
                
            except Exception as e:
                logger.error(f"Error getting policy information: {str(e)}", exc_info=True)
                return {
                    'type': 'error',
                    'message_id': message_id,
                    'response': "I apologize, but I'm having trouble accessing the policy information right now. Please try again later or contact a healthcare provider directly for the most accurate information."
                }
        
        return None

    def _format_policy_response(self, policy_details, state_context):
        """
        Format policy information into a user-friendly response with proper citations
        
        Args:
            policy_details: The policy details for the state
            state_context: The state code or name
        
        Returns:
            dict: Formatted response with citation information
        """
        message_id = str(uuid.uuid4())
        
        if not policy_details:
            response_text = (
                f"I couldn't find specific healthcare providers in your area. For assistance finding a provider, contact "
                f"Planned Parenthood at 1-800-230-PLAN. **Additional Resources:** - Planned Parenthood: 1-800-230-PLAN - "
                f"National Abortion Federation Hotline: 1-800-772-9100 - www.ineedana.com - Find local resources and clinics"
            )
        else:
            # Format policy response based on the provided details
            response_text = f"**Policy Information for {state_context}:** {policy_details}"
        
        # Create citations list with proper source links
        citations = [
            {
                "source": "Abortion Policy API",
                "url": "https://www.abortionpolicyapi.com/"
            },
            {
                "source": "Planned Parenthood",
                "url": "https://www.plannedparenthood.org/learn/abortion"
            },
            {
                "source": "National Abortion Federation",
                "url": "https://prochoice.org/"
            }
        ]
        
        # Get citation objects for formatting in frontend
        citation_objects = [
            {
                "source": "Abortion Policy API",
                "url": "https://www.abortionpolicyapi.com/",
                "title": "Abortion Policy API",
                "authors": []
            },
            {
                "source": "Planned Parenthood",
                "url": "https://www.plannedparenthood.org/learn/abortion",
                "title": "Abortion Information",
                "authors": ["Planned Parenthood Federation of America"]
            },
            {
                "source": "National Abortion Federation",
                "url": "https://prochoice.org/",
                "title": "Find Abortion Care",
                "authors": ["National Abortion Federation"]
            }
        ]
        
        # Format the response
        response = {
            'text': response_text,
            'citations': [c["source"] for c in citations],
            'citation_objects': citation_objects,
            'message_id': message_id
        }
        
        return response

    def _handle_emotional_query(self, message, message_id):
        """
        Handle messages that express emotional needs, especially related to abortion stress
        
        Args:
            message (str): User message with emotional content
            message_id (str): Unique message identifier
            
        Returns:
            dict: Response with appropriate emotional support
        """
        logger.info("Handling emotional support message")
        
        # Add to history
        self.add_to_history('user', message)
        
        # Check for abortion-related emotional content
        abortion_related = any(term in message.lower() for term in ["abort", "abortion", "pregnancy", "pregnant", "terminate"])
        
        # If abortion-related and emotional, provide specific support
        if abortion_related:
            response_text = (
                "I understand that thinking about abortion can be stressful and emotional. " 
                "It's completely normal to feel overwhelmed, anxious, or uncertain during this time. " 
                "Your feelings are valid, and many people experience similar emotions when making decisions about pregnancy. "
                "\n\nIf you'd like to talk to someone directly, Planned Parenthood offers counseling services at 1-800-230-PLAN, "
                "and the All-Options Talkline (1-888-493-0092) provides judgment-free emotional support. "
                "\n\nIs there any specific information about abortion procedures or options that might help you right now?"
            )
        else:
            # General emotional support
            response_text = (
                "I hear that you're feeling stressed or emotional right now. That's completely understandable, "
                "and your feelings are valid. If you'd like to talk about anything specific related to reproductive health, "
                "I'm here to help with information. For immediate emotional support, you can also contact the "
                "All-Options Talkline at 1-888-493-0092 for judgment-free counseling."
            )
        
        response = {
            'text': response_text,
            'citations': ["All-Options Talkline", "Planned Parenthood"],
            'citation_objects': [
                {
                    'source': "All-Options Talkline",
                    'url': "https://www.all-options.org/find-support/talkline/",
                },
                {
                    'source': "Planned Parenthood",
                    'url': "https://www.plannedparenthood.org/learn/abortion/considering-abortion",
                }
            ],
            'message_id': message_id
        }
        
        # Add response to history
        self.add_to_history('bot', response_text, message_id=message_id)
        
        return response

    def _handle_multi_faceted_query(self, message, message_id, classification):
        """
        Handle messages that combine multiple types of queries (policy, RAG, emotional)
        
        Args:
            message (str): User message with multiple query types
            message_id (str): Unique message identifier
            classification (dict): Question classification results
            
        Returns:
            dict: Combined response addressing all query aspects
        """
        logger.info("Handling multi-faceted query with multiple components")
        
        # Add to conversation history
        self.add_to_history('user', message)
        
        response_parts = []
        citations = []
        citation_objects = []
        
        # Check for policy component
        if classification.get("is_policy_question", False):
            logger.info("Processing policy component of multi-faceted query")
            # Extract state/location context
            state_context = None
            zip_code = self._extract_zip_code(message)
            
            if zip_code:
                state_context = self._get_state_from_zip_code(zip_code)
            else:
                state_context = self._check_for_state_names(message)
                
            if state_context:
                # Get policy response but don't return it yet
                policy_response = self._get_policy_response_for_state(state_context, message_id, zip_code)
                if policy_response:
                    response_parts.append("**Abortion Policy Information:**\n" + policy_response['text'])
                    citations.extend(policy_response.get('citations', []))
                    citation_objects.extend(policy_response.get('citation_objects', []))
        
        # Check for knowledge base / information need
        if "information" in classification.get("categories", []) or classification.get("is_abortion_related", False):
            logger.info("Processing information component of multi-faceted query")
            # Use BERT RAG model for reproductive health information
            from chatbot.bert_rag import BertRAGModel
            rag_model = BertRAGModel()
            
            # Get information response
            info_response = rag_model.get_response(message)
            if info_response and "I'm not sure I understand" not in info_response:
                response_parts.append("**Health Information:**\n" + info_response)
                # Extract citations if they exist in the RAG response
                if "Source:" in info_response:
                    citations.append("Planned Parenthood")
                    citation_objects.append({
                        'source': "Planned Parenthood",
                        'url': "https://www.plannedparenthood.org/learn",
                    })
        
        # Check for emotional support need
        if classification.get("is_emotional", False):
            logger.info("Processing emotional support component of multi-faceted query")
            # Check for abortion-related emotional content
            abortion_related = any(term in message.lower() for term in ["abort", "abortion", "pregnancy", "pregnant", "terminate"])
            
            # Add emotional support response
            if abortion_related:
                emotional_text = (
                    "**Emotional Support:**\n"
                    "I understand that thinking about abortion can be stressful and emotional. "
                    "It's completely normal to feel overwhelmed during this time. "
                    "Planned Parenthood offers counseling at 1-800-230-PLAN, and the All-Options Talkline (1-888-493-0092) provides judgment-free support."
                )
            else:
                emotional_text = (
                    "**Emotional Support:**\n"
                    "I hear that you're feeling stressed or emotional. That's completely understandable. "
                    "For immediate emotional support, you can contact the All-Options Talkline at 1-888-493-0092 for judgment-free counseling."
                )
                
            response_parts.append(emotional_text)
            
            # Add emotional support citations
            if "All-Options" not in [c.get('source') for c in citation_objects if c.get('source')]:
                citations.append("All-Options Talkline")
                citation_objects.append({
                    'source': "All-Options Talkline",
                    'url': "https://www.all-options.org/find-support/talkline/",
                })
        
        # Combine all response parts
        combined_response = "\n\n".join(response_parts)
        
        # Create final response
        response = {
            'text': combined_response,
            'citations': list(set(citations)),  # Remove duplicates
            'citation_objects': citation_objects,
            'message_id': message_id
        }
        
        # Add response to history
        self.add_to_history('bot', combined_response, message_id=message_id)
        
        return response

    def _is_emergency_contraception_query(self, message):
        """
        Determine if a message is asking about emergency contraception
        
        Args:
            message (str): User message text
            
        Returns:
            bool: True if the message is about emergency contraception, False otherwise
        """
        # Keywords related to emergency contraception
        ec_keywords = [
            'emergency contraception', 'plan b', 'morning after', 'ella', 
            'morning-after pill', 'day after pill', 'contraceptive pill',
            'morning pill', 'after pill', 'emergency pill', 'morning after pill',
            'plan b pill', 'iud as emergency', 'copper iud emergency', 'emergency iud',
            'what are the different types of emergency contraception', 'types of emergency contraception',
            'kinds of emergency contraception', 'what kinds of emergency contraception', 
            'what types of emergency contraception', 'emergency birth control',
            'contraception after sex', 'birth control after sex', 'after sex pill'
        ]
        
        message_lower = message.lower()
        
        # Check for emergency contraception keywords
        for keyword in ec_keywords:
            if keyword in message_lower:
                logger.info(f"Emergency contraception query detected: contains '{keyword}'")
                return True
                
        # Check for combination of terms
        emergency_terms = ['emergency', 'morning after', 'day after', 'plan b', 'after sex', 'after unprotected']
        contraception_terms = ['contraception', 'birth control', 'pill', 'iud', 'prevent pregnancy']
        
        has_emergency = any(term in message_lower for term in emergency_terms)
        has_contraception = any(term in message_lower for term in contraception_terms)
        
        if has_emergency and has_contraception:
            logger.info("Emergency contraception query detected: combined emergency and contraception terms")
            return True
            
        return False
        
    def _get_emergency_contraception_response(self, message, message_id):
        """
        Get response for emergency contraception questions
        
        Args:
            message (str): User message text
            message_id (str): ID for this message
            
        Returns:
            dict: Response data structure with text and metadata
        """
        logger.info("Retrieving emergency contraception information")
        
        # Add message to history
        self.add_to_history('user', message)
        
        try:
            # RAG approach to get most relevant emergency contraception information
            message_lower = message.lower()
            
            # Get relevant information from our reproductive health data
            ec_response = ""
            
            # If asking about types/kinds of emergency contraception
            if any(term in message_lower for term in ['what kind', 'what type', 'different types', 'different kinds', 'types of', 'kinds of']):
                ec_response = (
                    "There are 2 main types of emergency contraception:\n\n"
                    "1. Emergency Contraception Pills (Morning-after pills):\n"
                    "• Plan B One-Step, Next Choice One Dose, and similar pills that contain levonorgestrel\n"
                    "  - Most widely available, no prescription needed for any age\n"
                    "  - Works best within 72 hours (3 days) but can be taken up to 5 days\n"
                    "  - More effective for people weighing less than 165 pounds\n"
                    "  - Available at pharmacies, grocery stores, and online ($40-$50)\n"
                    "• ella (ulipristal acetate) prescription pill\n"
                    "  - More effective than Plan B, especially for people weighing 165-195 pounds\n"
                    "  - Effective for up to 5 days (120 hours) after unprotected sex\n"
                    "  - Requires a prescription\n"
                    "  - Available at pharmacies with prescription ($50-$60)\n\n"
                    "2. Copper IUD (ParaGard) or hormonal IUDs (Mirena, Liletta):\n"
                    "• Most effective type of emergency contraception (over 99% effective) when inserted within 5 days\n"
                    "• Provides ongoing contraception for 8-12 years depending on type\n"
                    "• Requires insertion by a healthcare provider\n"
                    "• Cost varies ($0-$1,300 depending on insurance)\n\n"
                    "Emergency contraception works best when used as soon as possible after unprotected sex, though some methods are effective up to 5 days (120 hours) afterward."
                )
            # If asking how emergency contraception works
            elif any(term in message_lower for term in ['how does', 'how do', 'work', 'mechanism', 'effective']):
                ec_response = (
                    "Emergency contraception primarily works by preventing or delaying ovulation (the release of an egg from the ovary). It won't work if ovulation has already occurred.\n\n"
                    "• Emergency contraception pills (like Plan B) prevent pregnancy by temporarily stopping the release of an egg from the ovary\n"
                    "• The copper IUD (ParaGard) works by making sperm less able to fertilize an egg and may also prevent implantation\n\n"
                    "Key facts about how emergency contraception works:\n"
                    "• Emergency contraception is NOT an abortion pill - it prevents pregnancy before it starts\n"
                    "• It will not terminate an existing pregnancy\n"
                    "• It's safe to use with very few side effects (possibly nausea, headache, fatigue, irregular bleeding)\n"
                    "• It's a backup method and not as effective as regular contraception\n"
                    "• It does not protect against STIs\n\n"
                    "Important: Emergency contraception is not an abortion. It prevents pregnancy from occurring and won't end an existing pregnancy."
                )
            # If asking about timing or effectiveness
            elif any(term in message_lower for term in ['how long', 'when', 'too late', 'timing', 'effectiveness', 'effective']):
                ec_response = (
                    "Timing for emergency contraception:\n\n"
                    "• You have up to 5 days (120 hours) after unprotected sex to use emergency contraception, but sooner is better\n"
                    "• Plan B and similar pills work best within 72 hours (3 days)\n"
                    "• ella remains equally effective throughout the 5-day window\n"
                    "• Copper IUD (ParaGard) or hormonal IUDs (Mirena, Liletta) can be inserted up to 5 days after unprotected sex\n\n"
                    "Effectiveness:\n"
                    "• Copper IUD: More than 99% effective\n"
                    "• ella: About 85% effective\n"
                    "• Plan B: About 75-89% effective when taken within 72 hours\n\n"
                    "Emergency contraception becomes less effective if you weigh more than 165 pounds (75 kg). For people over 165 pounds, ella or a copper IUD may be more effective options."
                )
            # General information
            else:
                ec_response = (
                    "Emergency contraception helps prevent pregnancy after unprotected sex or when birth control fails. Key points:\n\n"
                    "• It works by delaying or preventing ovulation\n"
                    "• It should be used as soon as possible after unprotected sex (up to 5 days)\n"
                    "• It's not intended for regular use as birth control\n"
                    "• It doesn't protect against STIs\n"
                    "• It's not the same as abortion—it won't end an existing pregnancy\n\n"
                    "Types include:\n"
                    "• Plan B (over-the-counter) - available without prescription at pharmacies\n"
                    "• ella (prescription only) - more effective for people over 165 pounds\n"
                    "• Copper IUDs (requires healthcare provider insertion) - most effective option\n\n"
                    "Where to get emergency contraception:\n"
                    "• Pharmacies (Plan B is available over-the-counter)\n"
                    "• Healthcare providers (for prescriptions or IUD insertion)\n"
                    "• Planned Parenthood health centers\n"
                    "• University health centers\n"
                    "• Some can be ordered online through services like Nurx or SimpleHealth\n\n"
                    "For more information or personalized advice, please contact your healthcare provider or call Planned Parenthood at 1-800-230-PLAN."
                )
            
            # Format the response with sources
            formatted_response = {
                "text": ec_response,
                "citations": ["Planned Parenthood", "American College of Obstetricians and Gynecologists"],
                "citation_objects": [
                    {
                        'source': "Planned Parenthood",
                        'url': "https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception",
                    },
                    {
                        'source': "American College of Obstetricians and Gynecologists",
                        'url': "https://www.acog.org/womens-health/faqs/emergency-contraception",
                    }
                ],
                "message_id": message_id
            }
            
            # Add bot response to history
            self.add_to_history('bot', ec_response, message_id=message_id)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing emergency contraception question: {str(e)}")
            # Fallback response
            fallback_response = (
                "I can provide information about emergency contraception, which includes methods like Plan B and IUDs "
                "that can prevent pregnancy after unprotected sex. Could you please specify what you'd like to know about emergency contraception?"
            )
            
            formatted_response = {
                "text": fallback_response,
                "citations": ["Planned Parenthood"],
                "citation_objects": [
                    {
                        'source': "Planned Parenthood",
                        'url': "https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception",
                    }
                ],
                "message_id": message_id
            }
            
            # Add bot response to history
            self.add_to_history('bot', fallback_response, message_id=message_id)
            
            return formatted_response

    def _get_baseline_response(self, message, message_id):
        """
        Get a response from the baseline model
        
        Args:
            message (str): User message
            message_id (str): Message ID for tracking
            
        Returns:
            dict: Response with text and metadata
        """
        try:
            # Process the message through baseline categorization
            category = self.baseline_model.categorize_question(message)
            
            # Use appropriate handler based on category
            if category == "policy":
                # Policy questions should be handled by policy API, not here
                fallback_text = "I can help with information about abortion policies. Could you please specify which state you're asking about?"
                return {
                    "text": fallback_text,
                    "citations": [],
                    "citation_objects": [],
                    "message_id": message_id
                }
            elif category == "conversational" and self.question_classifier.classify_question(message).get("is_emotional", False):
                # Only use GPT for emotional support responses
                gpt_response = self.baseline_model.gpt_model.get_response(message)
                return {
                    "text": gpt_response,
                    "citations": [],
                    "citation_objects": [],
                    "message_id": message_id
                }
            else:
                # Use BERT RAG for all knowledge questions and non-emotional conversational questions
                rag_response = self.baseline_model.bert_rag.get_response(message)
                return {
                    "text": rag_response,
                    "citations": ["Planned Parenthood"],
                    "citation_objects": [{
                        "source": "Planned Parenthood",
                        "url": "https://www.plannedparenthood.org/learn"
                    }],
                    "message_id": message_id
                }
        except Exception as e:
            logger.error(f"Error generating baseline response: {str(e)}", exc_info=True)
            return {
                "text": "I'm sorry, I encountered an error processing your question. Could you try asking in a different way?",
                "citations": [],
                "citation_objects": [],
                "message_id": message_id
            }

    def _has_abortion_context_in_history(self):
        """
        Check if there is abortion-related context in the conversation history
        
        Returns:
            bool: True if abortion context exists, False otherwise
        """
        # Look through the last 5 messages in history for abortion-related content
        recent_history = self.conversation_history[-5:] if len(self.conversation_history) > 0 else []
        
        for entry in recent_history:
            if entry.get('sender') == 'user' and isinstance(entry.get('message'), str):
                user_message = entry.get('message', '').lower()
                # Check if this message was abortion-related
                if self._is_abortion_related(user_message):
                    logger.debug(f"Found abortion context in history: '{user_message}'")
                    return True
                
                # Check for policy or law related terms
                policy_terms = ['policy', 'policies', 'law', 'laws', 'legal', 'illegal', 'banned', 'ban', 'allowed', 'restrict', 'get an abortion']
                if any(term in user_message for term in policy_terms):
                    logger.debug(f"Found policy context in history: '{user_message}'")
                    return True
            
            # Also check if the bot's previous responses were about abortion policies
            if entry.get('sender') == 'bot' and isinstance(entry.get('message'), str):
                bot_message = entry.get('message', '').lower()
                if 'abortion policy' in bot_message or 'abortion is legal' in bot_message or 'abortion is banned' in bot_message or 'weeks of pregnancy' in bot_message:
                    logger.debug("Found abortion policy context in bot's previous response")
                    return True
                
        return False