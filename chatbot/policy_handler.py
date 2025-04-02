import logging
import os
import asyncio
import aiohttp
import json
import time
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import for OpenAI
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# Import for ZIP code lookup - optional, will use fallback if not available
try:
    from pyzipcode import ZipCodeDatabase
except ImportError:
    ZipCodeDatabase = None

logger = logging.getLogger(__name__)

class PolicyHandler:
    """
    Handles policy-related queries with location context awareness
    """
    
    def __init__(self, api_key: Optional[str] = None, policy_api_base_url: Optional[str] = None):
        """
        Initialize the policy handler
        
        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            policy_api_base_url (Optional[str]): Base URL for the abortion policy API
        """
        logger.info("Initializing PolicyHandler")
        
        # Load environment variables or use provided values
        self.openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.abortion_policy_api_key = os.environ.get("ABORTION_POLICY_API_KEY", "")  # Default empty string
        self.policy_api_base_url = policy_api_base_url or os.environ.get("POLICY_API_BASE_URL", "https://api.abortionpolicyapi.com/v1/")
        self.openai_client = None
        
        if self.openai_api_key and AsyncOpenAI:
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized for policy handler")
        else:
            logger.warning("OpenAI client not available, using template-based fallback")
        
        # States mapping - include full state names for better matching
        self.STATE_NAMES = {
            "AL": "Alabama",
            "AK": "Alaska",
            "AZ": "Arizona",
            "AR": "Arkansas",
            "CA": "California",
            "CO": "Colorado",
            "CT": "Connecticut",
            "DE": "Delaware",
            "FL": "Florida",
            "GA": "Georgia",
            "HI": "Hawaii",
            "ID": "Idaho",
            "IL": "Illinois",
            "IN": "Indiana",
            "IA": "Iowa",
            "KS": "Kansas",
            "KY": "Kentucky",
            "LA": "Louisiana",
            "ME": "Maine",
            "MD": "Maryland",
            "MA": "Massachusetts",
            "MI": "Michigan",
            "MN": "Minnesota",
            "MS": "Mississippi",
            "MO": "Missouri",
            "MT": "Montana",
            "NE": "Nebraska",
            "NV": "Nevada",
            "NH": "New Hampshire",
            "NJ": "New Jersey",
            "NM": "New Mexico",
            "NY": "New York",
            "NC": "North Carolina",
            "ND": "North Dakota",
            "OH": "Ohio",
            "OK": "Oklahoma",
            "OR": "Oregon",
            "PA": "Pennsylvania",
            "RI": "Rhode Island",
            "SC": "South Carolina",
            "SD": "South Dakota",
            "TN": "Tennessee",
            "TX": "Texas",
            "UT": "Utah",
            "VT": "Vermont",
            "VA": "Virginia",
            "WA": "Washington",
            "WV": "West Virginia",
            "WI": "Wisconsin",
            "WY": "Wyoming",
            "DC": "District of Columbia"
        }
        
        # Create lowercase state names for searches
        self.STATE_NAMES_LOWER = {k.lower(): k for k in self.STATE_NAMES.keys()}
        for k, v in self.STATE_NAMES.items():
            self.STATE_NAMES_LOWER[v.lower()] = k

        # Initialize cache for policy API responses
        self.policy_cache = {}
        self.cache_ttl = 86400  # 24 hours - policy data doesn't change often
        
        # Keep track of state associations for sessions
        self.session_state_cache = {}  # Maps session_id -> state_code
        
        # Endpoints for the Abortion Policy API
        self.api_base_url = self.policy_api_base_url
        self.api_endpoints = {
            "waiting_periods": "waiting_periods",
            "insurance_coverage": "insurance_coverage",
            "gestational_limits": "gestational_limits",
            "minors": "minors"
        }
    
    async def process_query(self, 
                         query: str, 
                         full_message: str = None, 
                         conversation_history: List[Dict] = None,
                         user_location: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a policy query to get state-specific abortion policy information
        
        Args:
            query (str): The aspect query to process
            full_message (str): The full original user message
            conversation_history (List[Dict]): List of previous conversation messages
            user_location (Optional[Dict[str, str]]): User's location information
            
        Returns:
            Dict[str, Any]: Policy response data
        """
        logger.info(f"Processing policy query: '{query}'...")
        
        # Initialize response
        state_code = None
        response_data: Dict[str, Any] = {}
        session_id = None
        
        # Extract session ID from conversation history
        if conversation_history:
            for msg in reversed(conversation_history):
                if isinstance(msg, dict) and msg.get("session_id"):
                    session_id = msg["session_id"]
                    logger.info(f"Found session ID: {session_id}")
                    break
        
        # Check for multiple states in the query
        state_mentions = self._get_all_state_mentions(query, full_message if full_message else query)
        
        # If we found multiple states, handle it as a comparison
        if len(state_mentions) > 1:
            logger.info(f"Multiple states detected in query: {state_mentions}")
            return await self._handle_state_comparison(query, state_mentions, full_message)
            
        # If we found exactly one state, use it directly
        elif len(state_mentions) == 1:
            state_code = state_mentions[0]
            logger.info(f"Found state name '{state_code}' in query")
            
            # Cache this state for future interactions in the same session
            if session_id:
                self.session_state_cache[session_id] = state_code
                logger.info(f"Cached state {state_code} for session {session_id}")
                
        # Otherwise proceed with previous logic for "my state" type queries
        else:
            # Check if we have a cached state for this session
            if session_id and session_id in self.session_state_cache:
                # Use the cached state
                state_code = self.session_state_cache[session_id]
                logger.info(f"Using cached state {state_code} for session {session_id}")
                
            else:
                # No cached state, need to determine from the query or other sources
                if session_id:
                    logger.info(f"Session {session_id} has no cached state yet")
                
                # 1. Check for explicit state mentions in the query
                logger.info(f"Checking for state mentions in query: '{query}'")
                
                # Handle "my state" type queries first
                if re.search(r'\b(my|our)\s+state\b', query.lower()):
                    logger.info("Query contains 'my state' mention")
                    
                    # Check if we have a location stored in the conversation
                    if conversation_history:
                        state_code = self._get_state_from_conversation(conversation_history)
                    
                    # If still no state, check user location
                    if not state_code and user_location:
                        state_code = self._get_state_from_location(user_location)
                        
                    if state_code:
                        logger.info(f"Resolved 'my state' to {state_code}")
                    else:
                        # Instead of proceeding with a default state, return a response asking for the state
                        logger.info("Could not resolve 'my state', sending response to ask user")
                        return {
                            "aspect_type": "policy",
                            "primary_content": "I'll need to know which state you're in to answer that question. Could you please tell me your state, so I can provide accurate information about abortion policies that apply to you?",
                            "question_answered": False,
                            "needs_state_info": True
                        }
                else:
                    logger.info("Regular query without 'my state' mention")
                    
                    # Try to get state from query directly
                    state_code = self._get_state_from_query(query=full_message if full_message else query)
                    
                    # Also check if ZIP code is mentioned and try to get state from that
                    if not state_code:
                        state_code = self._get_state_from_zip(query=full_message if full_message else query)
                
                # 2. If no state in query, check user's location data
                if not state_code:
                    logger.info("No cached state, checking user location and history...")
                    
                    # First check if we know user's location
                    if user_location:
                        state_code = self._get_state_from_location(user_location)
                        
                    # Next, look in conversation history for previous state references
                    if not state_code and conversation_history:
                        state_code = self._get_state_from_conversation(conversation_history)
                        
                    if state_code:
                        logger.info(f"✓ Found state {state_code} from location or history")
                    else:
                        logger.info("✗ No state found in user location or history")
        
        # Process based on whether we have a state or not
        if state_code:
            # Cache this state for future interactions in the same session
            if session_id:
                self.session_state_cache[session_id] = state_code
                logger.info(f"Cached state {state_code} for session {session_id}")
            
            # Get policy data for this state
            policy_data = await self._fetch_policy_data(state_code)
            
            if policy_data:
                logger.info(f"Retrieved policy data for {state_code}")
                logger.info(f"Policy data structure keys: {policy_data.keys()}")
                
                # Add policy data to response
                response_data = {
                    "aspect_type": "policy",
                    "primary_content": "",  # Will be filled below
                    "state_code": state_code,
                    "state_name": self.STATE_NAMES.get(state_code, ""),
                    "question_answered": True
                }
                
                # Create basic summary from the endpoints data
                state_name = self.STATE_NAMES.get(state_code, state_code)
                summary = f"Here's abortion policy information for {state_name}:\n\n"
                
                policy_details = {}
                
                # Process each endpoint's data
                for endpoint_name, endpoint_data in policy_data.get("endpoints", {}).items():
                    logger.info(f"Processing endpoint {endpoint_name} with data: {endpoint_data}")
                    
                    # Handle nested state structure (like {'Maine': {...}})
                    if endpoint_data and isinstance(endpoint_data, dict):
                        state_specific_data = None
                        
                        # Check if we need to extract state-specific data
                        if state_name in endpoint_data:
                            state_specific_data = endpoint_data.get(state_name)
                            logger.info(f"Found nested state data for {state_name}: {state_specific_data}")
                        elif state_code in endpoint_data:
                            state_specific_data = endpoint_data.get(state_code)
                            logger.info(f"Found nested state data for {state_code}: {state_specific_data}")
                            
                        # If we found nested state data, use that instead
                        if state_specific_data and isinstance(state_specific_data, dict):
                            endpoint_data = state_specific_data
                            
                        # Convert endpoint name to readable format
                        readable_endpoint = endpoint_name.replace("_", " ").title()
                        
                        # Extract relevant data points based on endpoint type
                        if endpoint_name == "gestational_limits":
                            if endpoint_data.get("banned", False):
                                summary += f"• Abortion is banned in {state_name}.\n"
                                policy_details["legal_status"] = "Banned"
                            elif "banned_after_weeks_since_LMP" in endpoint_data:
                                weeks = endpoint_data["banned_after_weeks_since_LMP"]
                                # If weeks is 99, it usually means no limit or viability
                                if weeks == 99:
                                    summary += f"• Abortion is legal in {state_name} until viability or with medical necessity.\n"
                                    policy_details["gestational_limit"] = "Until viability"
                                else:
                                    summary += f"• Abortion is prohibited after {weeks} weeks since last menstrual period.\n"
                                    policy_details["gestational_limit"] = f"{weeks} weeks"
                                    
                            # Check for exceptions
                            exceptions = []
                            if endpoint_data.get("exception_life", False):
                                exceptions.append("to save the life of the pregnant person")
                                policy_details["life_exception"] = "Yes"
                            if endpoint_data.get("exception_health") == "Any":
                                exceptions.append("for health reasons")
                                policy_details["health_exception"] = "Yes"
                            elif endpoint_data.get("exception_health") == "Major Bodily Function":
                                exceptions.append("to prevent substantial impairment of bodily function")
                                policy_details["health_exception"] = "Limited"
                                
                            if endpoint_data.get("exception_rape_or_incest", False):
                                exceptions.append("in cases of rape or incest")
                                policy_details["rape_incest_exception"] = "Yes"
                                
                            if endpoint_data.get("exception_fetal", False):
                                exceptions.append("for fetal anomalies")
                                policy_details["fetal_anomaly_exception"] = "Yes"
                                
                            if exceptions:
                                summary += f"• Exceptions are allowed {', '.join(exceptions)}.\n"
                            
                        elif endpoint_name == "waiting_periods":
                            if "waiting_period_hours" in endpoint_data:
                                hours = endpoint_data["waiting_period_hours"]
                                if hours > 0:
                                    summary += f"• There is a required waiting period of {hours} hours.\n"
                                    policy_details["waiting_period"] = f"{hours} hours"
                                else:
                                    summary += f"• There is no required waiting period in {state_name}.\n"
                                    policy_details["waiting_period"] = "None"
                            
                        elif endpoint_name == "insurance_coverage":
                            coverage_points = []
                            
                            # Check for private insurance
                            if endpoint_data.get("private_coverage_prohibited", False):
                                coverage_points.append("Private insurance coverage for abortion is prohibited")
                                policy_details["private_insurance"] = "Prohibited"
                            elif endpoint_data.get("requires_coverage", True):
                                coverage_points.append("Private insurance must cover abortion services")
                                policy_details["private_insurance"] = "Required"
                                
                            # Check for exchange coverage
                            if endpoint_data.get("exchange_coverage_prohibited", False):
                                coverage_points.append("Insurance purchased through the health exchange cannot cover abortion")
                                policy_details["exchange_insurance"] = "Prohibited"
                                
                            # Check for Medicaid
                            if endpoint_data.get("medicaid_coverage_provider", "") == "yes" or endpoint_data.get("medicaid_coverage_provider_patient_decision", True):
                                coverage_points.append("Medicaid provides coverage for abortion in certain circumstances")
                                policy_details["medicaid_coverage"] = "Limited coverage"
                                
                            if coverage_points:
                                summary += f"• Insurance Coverage: {'; '.join(coverage_points)}.\n"
                            
                        elif endpoint_name == "minors":
                            minor_points = []
                            if endpoint_data.get("parental_consent_required", False):
                                minor_points.append("Parental consent is required")
                                policy_details["parental_consent"] = "Required"
                                
                                # Check for additional context that might override
                                additional_context = endpoint_data.get("additional_context", "")
                                if "able to consent without a parent" in additional_context or "minors are able to consent" in additional_context:
                                    logger.info(f"Found override for parental consent: {additional_context}")
                                    minor_points = ["Minors may be able to consent without parental involvement under certain circumstances"]
                                    policy_details["parental_consent"] = "Limited exceptions available"
                            
                            if endpoint_data.get("parental_notification_required", False):
                                minor_points.append("Parents must be notified")
                                policy_details["parental_notification"] = "Required"
                                
                            if minor_points:
                                summary += f"• Minors: {'; '.join(minor_points)}.\n"
                        
                        # Handle any other endpoints or data
                        elif endpoint_data:
                            # Extract key information
                            data_points = []
                            for key, value in endpoint_data.items():
                                if key not in ["id", "state", "created_at", "updated_at", "Last Updated"] and value:
                                    if isinstance(value, bool):
                                        data_points.append(f"{key.replace('_', ' ')}: {'Yes' if value else 'No'}")
                                    else:
                                        data_points.append(f"{key.replace('_', ' ')}: {value}")
                            
                            if data_points:
                                summary += f"• {readable_endpoint}: {'; '.join(data_points[:3])}.\n"
                                policy_details[endpoint_name.replace('_', '_')] = ', '.join(data_points[:3])
                    elif endpoint_data:
                        # Handle non-dict data (strings, etc.)
                        summary += f"• {endpoint_name.replace('_', ' ').title()}: {endpoint_data}\n"
                
                # If we don't have any policy details yet, create a generic entry
                if not policy_details and state_name:
                    summary += f"• Abortion is currently regulated in {state_name}. Please check with official sources for specific details.\n"
                    policy_details["note"] = f"Limited data available for {state_name}"
                
                # Add resources and disclaimer
                summary += "\nThis information is based on the most recent data available, but laws may have changed. "
                summary += "For the most up-to-date information, please contact Planned Parenthood or visit abortionfinder.org."
                
                # Add the generated summary and details to the response
                response_data["primary_content"] = summary
                response_data["policy_details"] = policy_details
                response_data["policy_url"] = "https://www.abortionpolicyapi.com/"
                response_data["policy_last_updated"] = datetime.now().strftime('%Y-%m-%d')
                response_data["supportive_resources"] = self._get_supportive_resources_list(state_code)
                
                # Ensure we have at least some content in the response
                if not summary.strip() or summary.strip() == "Here's abortion policy information for " + state_name + ":\n\n":
                    # Generate a basic summary if we couldn't extract specific details
                    logger.info(f"No specific policy details extracted for {state_code}, generating basic info")
                    basic_summary = f"Here's abortion policy information for {state_name}:\n\n"
                    basic_summary += f"• For detailed information about abortion policies in {state_name}, please refer to official sources.\n"
                    basic_summary += f"• Abortion laws vary by state and may change - always consult trusted healthcare providers.\n\n"
                    basic_summary += "For the most up-to-date information, please contact Planned Parenthood or visit abortionfinder.org."
                    response_data["primary_content"] = basic_summary
            else:
                logger.warning(f"Failed to retrieve policy data for {state_code}")
                response_data = {
                    "aspect_type": "policy",
                    "primary_content": f"I'm sorry, but I couldn't retrieve the abortion policy information for {self.STATE_NAMES.get(state_code, state_code)} at this time. Please try again later or check official sources for the most accurate information.",
                    "state_code": state_code,
                    "state_name": self.STATE_NAMES.get(state_code, ""),
                    "question_answered": False
                }
        else:
            # If we can't determine a state and it's clearly a state-specific query,
            # ask the user to specify their state instead of using a default
            logger.warning("No state identified for policy query, asking user to specify")
            
            # Look for policy-related keywords that suggest we need state context
            policy_keywords = ["abortion", "reproductive", "contraception", "birth control", 
                              "pregnancy", "termination", "law", "policy", "legal"]
            
            has_policy_terms = any(keyword in query.lower() for keyword in policy_keywords)
            
            if has_policy_terms:
                return {
                    "aspect_type": "policy",
                    "primary_content": "To answer your question about abortion or reproductive health policy, I'll need to know which state you're in. Laws vary significantly by state. Could you please tell me which state you're asking about?",
                    "question_answered": False,
                    "needs_state_info": True
                }
            else:
                # If it's not clearly a policy query, provide a general response
                return {
                    "aspect_type": "policy",
                    "primary_content": "I'm not sure which state's policy you're asking about. Different states have very different laws regarding abortion and reproductive healthcare. Could you please clarify which state you're interested in?",
                    "question_answered": False,
                    "needs_state_info": True
                }
        
        return response_data
    
    def _get_all_state_mentions(self, query: str, full_message: str) -> List[str]:
        """
        Get all state mentions in a query (for handling multi-state queries)
        
        Args:
            query (str): The query text
            full_message (str): The full message text
            
        Returns:
            List[str]: List of state codes found
        """
        message_text = full_message.lower() if full_message else query.lower()
        state_mentions = []
        
        # First look for full state names
        for state_code, state_name in self.STATE_NAMES.items():
            state_pattern = r'\b' + re.escape(state_name.lower()) + r'\b'
            if re.search(state_pattern, message_text):
                state_mentions.append(state_code)
                logger.info(f"Found state name mention: {state_name}")
        
        # Then look for state codes
        for state_code in self.STATE_NAMES.keys():
            # Skip "in" as it's commonly used as a preposition
            if state_code.lower() == "in":
                continue
                
            code_pattern = r'\b' + re.escape(state_code.lower()) + r'\b'
            if re.search(code_pattern, message_text) and state_code not in state_mentions:
                # Double check problematic state codes that could match common words
                if state_code.lower() in ["or", "me", "hi", "ok", "de", "pa", "oh", "id", "co", "wa", "md", "va"]:
                    # Get context around the match to check if it's truly a state reference
                    match = re.search(rf'(\w+\s+)?\b{re.escape(state_code.lower())}\b(\s+\w+)?', message_text)
                    if match:
                        context = match.group(0)
                        if any(term in context for term in ["state", "states", "in", "laws", "abortion"]):
                            state_mentions.append(state_code)
                            logger.info(f"Found state code mention with context: {state_code} in '{context}'")
                else:
                    state_mentions.append(state_code)
                    logger.info(f"Found state code mention: {state_code}")
        
        # Look for ZIP codes
        zip_matches = re.findall(r'\b(\d{5})\b', message_text)
        for zip_code in zip_matches:
            state_from_zip = self._get_state_from_zip(zip_code)
            if state_from_zip and state_from_zip not in state_mentions:
                state_mentions.append(state_from_zip)
                logger.info(f"Found state from ZIP code {zip_code}: {state_from_zip}")
        
        # Remove duplicates while preserving order
        unique_states = []
        for state in state_mentions:
            if state not in unique_states:
                unique_states.append(state)
        
        return unique_states
    
    async def _handle_state_comparison(self, query: str, state_codes: List[str], full_message: str = None) -> Dict[str, Any]:
        """
        Handle comparison between multiple states
        
        Args:
            query (str): The query text
            state_codes (List[str]): List of state codes to compare
            full_message (str): The full message text
            
        Returns:
            Dict[str, Any]: Response with policy comparison
        """
        start_time = time.time()
        
        try:
            # Limit to max 3 states to avoid overload
            if len(state_codes) > 3:
                logger.info(f"Limiting comparison from {len(state_codes)} states to first 3")
                state_codes = state_codes[:3]
            
            # Fetch policy data for all states
            policy_data_dict = {}
            for state_code in state_codes:
                policy_data = await self._fetch_policy_data(state_code)
                policy_data_dict[state_code] = policy_data
            
            # Prepare state names for better readability
            state_names = [f"{self.STATE_NAMES.get(code, code)} ({code})" for code in state_codes]
            
            # Generate comparison text
            if self.openai_client:
                comparison_prompt = f"Compare abortion laws and policies in {', '.join(state_names)}. Highlight key differences in legality, gestational limits, and requirements.\n\n"
                
                # Add policy data for each state
                for state_code, policy_data in policy_data_dict.items():
                    comparison_prompt += f"\n{self.STATE_NAMES.get(state_code, state_code)} ({state_code}) policy data:\n"
                    for key, value in policy_data.items():
                        if key != "citations":
                            comparison_prompt += f"- {key}: {value}\n"
                
                # Generate the comparison using OpenAI
                messages = [
                    {"role": "system", "content": "You are a helpful assistant providing accurate information about abortion policies and laws in the United States. Present information in a clear, factual manner. Focus on key differences between states."},
                    {"role": "user", "content": comparison_prompt}
                ]
                
                response = await self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )
                
                comparison_text = response.choices[0].message.content.strip()
            else:
                # Fallback without OpenAI
                comparison_text = f"Comparison of abortion laws in {', '.join(state_names)}:\n\n"
                
                # Simple manual comparison of key factors
                factors = ["legal_status", "gestational_limits", "required_counseling", "waiting_period", "parental_consent_required"]
                
                for factor in factors:
                    comparison_text += f"\n{factor.replace('_', ' ').title()}:\n"
                    for state_code in state_codes:
                        state_name = self.STATE_NAMES.get(state_code, state_code)
                        value = policy_data_dict[state_code].get(factor, "Information not available")
                        comparison_text += f"- {state_name}: {value}\n"
            
            # Combine citations from all states
            all_citations = []
            for state_code in state_codes:
                citations = self._get_policy_citations(state_code)
                for citation in citations:
                    if citation not in all_citations:
                        all_citations.append(citation)
            
            # Measure processing time
            processing_time = time.time() - start_time
            logger.info(f"Policy comparison processed in {processing_time:.2f}s")
            
            # Prepare response object
            response = {
                "text": comparison_text,
                "aspect_type": "policy",
                "state_codes": state_codes,
                "confidence": 0.9,
                "citations": all_citations,
                "processing_time": processing_time
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing policy comparison: {str(e)}", exc_info=True)
            return {
                "text": f"I'm sorry, I'm having trouble comparing abortion policies between {', '.join([self.STATE_NAMES.get(code, code) for code in state_codes])}. Abortion laws vary by state and may change. You might consider contacting Planned Parenthood for the most current details.",
                "aspect_type": "policy",
                "confidence": 0.5,
                "citations": [{
                    "source": "Planned Parenthood",
                    "url": "https://www.plannedparenthood.org",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }]
            }
    
    def _get_state_from_query(self, query: str) -> Optional[str]:
        """
        Extract state name or code from the query
        
        Args:
            query (str): User query text
            
        Returns:
            Optional[str]: Two-letter state code or None if not found
        """
        # Check for common patterns involving state names
        # First, directly check for state names since they're most reliable
        for state_code, state_name in self.STATE_NAMES.items():
            # Look for full state name (case insensitive)
            if re.search(r'\b' + re.escape(state_name) + r'\b', query, re.IGNORECASE):
                logger.info(f"Found state name '{state_name}' in query")
                return state_code
            
            # Also check lowercase version
            if re.search(r'\b' + re.escape(state_name.lower()) + r'\b', query.lower()):
                logger.info(f"Found lowercase state name '{state_name.lower()}' in query")
                return state_code
        
        # Then check state codes (which are shorter and could match other words)
        for state_code in self.STATE_NAMES.keys():
            # Look for the state code (ensure it's a standalone word)
            if re.search(r'\b' + re.escape(state_code) + r'\b', query.upper()):
                # For problematic state codes that could match common words
                if state_code in ["IN", "OR", "ME", "HI", "OK", "DE", "PA", "OH"]:
                    # Make sure it's not part of a larger word or common phrase
                    # This is a simplified check - could be enhanced further
                    context = re.search(r'.{0,15}\b' + re.escape(state_code) + r'\b.{0,15}', query.upper())
                    if context:
                        context_str = context.group(0)
                        # Skip if it's used as a preposition in common phrases
                        if state_code == "IN" and any(phrase.upper() in context_str for phrase in ["IN MY", "IN THE", "IN YOUR", "IN A", "IN OUR", "LIVE IN"]):
                            logger.info(f"Skipping state code '{state_code}' as it appears to be the preposition 'in'")
                            continue
                
                logger.info(f"Found state code '{state_code}' in query")
                return state_code
        
        # Handle special cases for state name variations
        state_variants = {
            "maine": "ME",
            "main": "ME",
            "mayne": "ME",
            "texas": "TX",
            "new york": "NY",
            "ny": "NY",
            "cali": "CA",
            "fla": "FL",
            "mass": "MA",
            "penn": "PA"
            # Add more variants as needed
        }
        
        for variant, code in state_variants.items():
            if re.search(r'\b' + re.escape(variant) + r'\b', query.lower()):
                logger.info(f"Found state variant '{variant}' in query, matching to {code}")
                return code
        
        # If no state found
        return None
    
    def _get_state_from_history(self, user_location: Optional[Dict[str, str]], 
                               conversation_history: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """
        Extract state code from user location or conversation history
        
        Args:
            user_location (Optional[Dict[str, str]]): User's location data
            conversation_history (Optional[List[Dict[str, Any]]]): Conversation history
            
        Returns:
            Optional[str]: State code or None
        """
        # 1. Try user location first if provided
        if user_location:
            # Check for state code
            state_code = user_location.get('state', '').upper()
            if state_code and len(state_code) == 2 and state_code in self.STATE_NAMES:
                logger.info(f"Found state code in user location: {state_code}")
                return state_code
                
            # Check for state name
            state_name = user_location.get('state', '')
            for code, name in self.STATE_NAMES.items():
                if name.lower() == state_name.lower():
                    logger.info(f"Found state name in user location: {name}")
                    return code
            
            # Check for ZIP code in user location
            zip_code = user_location.get('zip', '') or user_location.get('postal_code', '')
            if zip_code and len(zip_code) >= 5:
                state_from_zip = self._get_state_from_zip(query=zip_code)
                if state_from_zip:
                    logger.info(f"Found state from ZIP code in user location: {state_from_zip}")
                    return state_from_zip
        
        # 2. Check conversation history
        if conversation_history:
            # Look at the last 5 messages from the user
            user_msgs = [msg.get('message', '').lower() 
                         for msg in conversation_history[-5:] 
                         if msg.get('sender') == 'user']
            
            for msg in user_msgs:
                # Try each message for state mentions
                state_found = self._get_state_from_query(query=msg)
                if state_found:
                    logger.info(f"Found state {state_found} in conversation history")
                    return state_found
                
                # Check for standalone state names or abbreviations in short messages
                if len(msg.split()) <= 3:  # Short message like "Texas" or "TX"
                    msg_cleaned = msg.strip().rstrip('.!?')
                    msg_upper = msg_cleaned.upper()
                    
                    # Check for exact state code
                    if len(msg_cleaned) == 2 and msg_upper in self.STATE_NAMES:
                        logger.info(f"Found standalone state code in history: {msg_upper}")
                        return msg_upper
                    
                    # Check for exact state name
                    for state_code, state_name in self.STATE_NAMES.items():
                        if state_name.lower() == msg_cleaned.lower():
                            logger.info(f"Found standalone state name in history: {state_name}")
                            return state_code
        
        # No state found
        return None
    
    async def _fetch_policy_data(self, state_code: str) -> Dict[str, Any]:
        """
        Fetch policy data from the Abortion Policy API
        
        Args:
            state_code (str): Two-letter state code
            
        Returns:
            Dict[str, Any]: Policy data
        """
        try:
            headers = {"token": self.abortion_policy_api_key}
            policy_info = {"endpoints": {}}
            
            # Ensure the API base URL doesn't end with a slash for consistent URL formatting
            api_base = self.api_base_url.rstrip('/')
            
            # Use aiohttp for asynchronous API calls
            async with aiohttp.ClientSession() as session:
                # Make multiple API calls in parallel
                tasks = []
                for key, endpoint in self.api_endpoints.items():
                    url = f"{api_base}/{endpoint}/states/{state_code}"
                    tasks.append(self._fetch_endpoint(session, url, headers, key))
                
                # Wait for all API calls to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for key, result in zip(self.api_endpoints.keys(), results):
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching {key} for {state_code}: {str(result)}")
                        policy_info["endpoints"][key] = {}
                    else:
                        policy_info["endpoints"][key] = result
            
            # If we have data, add state information
            if any(policy_info["endpoints"].values()):
                policy_info["state_code"] = state_code
                policy_info["state_name"] = self.STATE_NAMES.get(state_code, state_code)
                logger.info(f"Successfully fetched policy data for {state_code}")
                return policy_info
            else:
                logger.error(f"No policy data found for {state_code}")
                return self._get_fallback_policy_data(state_code)
                
        except Exception as e:
            logger.error(f"Error fetching policy data for {state_code}: {str(e)}", exc_info=True)
            return self._get_fallback_policy_data(state_code)
    
    async def _fetch_endpoint(self, session, url, headers, key):
        """
        Fetch data from a specific endpoint
        
        Args:
            session: aiohttp ClientSession
            url: API endpoint URL
            headers: Request headers
            key: Endpoint key
            
        Returns:
            dict: API response data
        """
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API returned status {response.status} for {url}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching endpoint {key}: {str(e)}")
            return {}
    
    def _get_fallback_policy_data(self, state_code: str) -> Dict[str, Any]:
        """
        Get fallback policy data when API fails
        
        Args:
            state_code (str): Two-letter state code
            
        Returns:
            Dict[str, Any]: Fallback policy data
        """
        state_name = self.STATE_NAMES.get(state_code, state_code)
        return {
            "state_code": state_code,
            "state_name": state_name,
            "error": True,
            "endpoints": {},
            "resources": [
                "Planned Parenthood",
                "National Abortion Federation",
                "INeedAnA.com"
            ],
            "sources": [
                {
                    "title": "Planned Parenthood",
                    "url": "https://www.plannedparenthood.org/",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ]
        }
    
    async def _generate_with_openai(self, query: str, state_code: str, 
                                   policy_data: Dict[str, Any]) -> str:
        """
        Generate a policy response using OpenAI
        
        Args:
            query (str): The user's query
            state_code (str): Two-letter state code
            policy_data (Dict[str, Any]): Policy data
            
        Returns:
            str: Generated response
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not available")
        
        # Convert policy data to a string
        policy_json = json.dumps(policy_data, indent=2)
        state_name = self.STATE_NAMES.get(state_code, state_code)
        
        system_message = f"""You are an expert reproductive health assistant specialized in abortion policy information.
Use the provided JSON data to answer questions about abortion policy in {state_name}.
Be accurate, factual, and concise. Specify any limitations or restrictions clearly.
If the data doesn't contain certain information, acknowledge that gap.
IMPORTANT: Focus ONLY on the provided data for {state_name}. Do not reference policies for other states.
Format your response in a clear, easy-to-understand way for someone seeking policy information."""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use the most advanced model for best quality
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Policy data for {state_name}:\n{policy_json}\n\nQuestion: {query}"}
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating policy response with OpenAI: {str(e)}", exc_info=True)
            # Fall back to template response
            return self._format_policy_data(state_code, policy_data)
    
    def _format_policy_data(self, state_code: str, policy_data: Dict[str, Any]) -> str:
        """
        Format policy data into a readable response
        
        Args:
            state_code (str): Two-letter state code
            policy_data (Dict[str, Any]): Policy data
            
        Returns:
            str: Formatted response
        """
        state_name = self.STATE_NAMES.get(state_code, state_code)
        
        # Check if we have an error
        if policy_data.get("error", False):
            return (
                f"I'm sorry, I'm having trouble accessing the latest policy information for {state_name}. "
                f"Abortion laws vary by state and may change frequently. "
                f"You might consider contacting Planned Parenthood or check abortionfinder.org for the most current details."
            )
        
        # Start building the response
        response = f"Here's the abortion policy information for {state_name}:\n\n"
        
        # Add gestational limits
        gestational_data = policy_data.get("endpoints", {}).get("gestational_limits", {})
        if gestational_data:
            if gestational_data.get("banned", False):
                response += f"Abortion is banned in {state_name}."
            elif "banned_after_weeks_since_LMP" in gestational_data:
                weeks = gestational_data["banned_after_weeks_since_LMP"]
                response += f"In {state_name}, abortion is prohibited after {weeks} weeks since last menstrual period."
            else:
                response += f"I don't have specific information about gestational limits in {state_name}."
        
        # Add waiting periods
        waiting_data = policy_data.get("endpoints", {}).get("waiting_periods", {})
        if waiting_data and "waiting_period_hours" in waiting_data:
            hours = waiting_data["waiting_period_hours"]
            if hours > 0:
                response += f" There is a required waiting period of {hours} hours."
        
        # Add insurance coverage information
        insurance_data = policy_data.get("endpoints", {}).get("insurance_coverage", {})
        if insurance_data:
            if insurance_data.get("private_coverage_prohibited", False):
                response += " Private insurance coverage for abortion is prohibited."
            if insurance_data.get("exchange_coverage_prohibited", False):
                response += " Insurance purchased through the health exchange cannot cover abortion."
            if insurance_data.get("medicaid_coverage_provider", "") == "yes":
                response += " Medicaid does provide coverage for abortion in certain circumstances."
        
        # Add minors restrictions
        minors_data = policy_data.get("endpoints", {}).get("minors", {})
        if minors_data:
            if minors_data.get("parental_consent_required", False):
                response += " Parental consent is required for minors seeking abortion."
            if minors_data.get("parental_notification_required", False):
                response += " Parents must be notified before a minor can obtain an abortion."
        
        # Add resources and disclaimer
        response += (
            "\n\nThis information is based on the most recent data available, but laws may have changed. "
            "For the most up-to-date information, please contact Planned Parenthood or visit abortionfinder.org."
        )
        
        return response
    
    def _get_policy_citations(self, state_code: str) -> List[Dict[str, Any]]:
        """
        Get citations for policy data
        
        Args:
            state_code (str): Two-letter state code
            
        Returns:
            List[Dict[str, Any]]: Citations
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        return [
            {
                "source": "Abortion Policy API",
                "url": "https://www.abortionpolicyapi.com/",
                "accessed_date": today
            },
            {
                "source": "Planned Parenthood",
                "url": "https://www.plannedparenthood.org/learn/abortion/abortion-laws-by-state",
                "accessed_date": today
            },
            {
                "source": "Abortion Finder",
                "url": "https://www.abortionfinder.org/",
                "accessed_date": today
            }
        ]
    
    def _get_state_from_zip(self, query: str) -> Optional[str]:
        """
        Extract state code from a ZIP code in the query using zipcodes library
        
        Args:
            query (str): The query text that might contain a ZIP code
            
        Returns:
            Optional[str]: State code or None
        """
        try:
            # Import Preprocessor here to avoid circular dependencies
            from .preprocessor import Preprocessor
            
            # Check if we have initialized a preprocessor
            if not hasattr(self, '_preprocessor'):
                self._preprocessor = Preprocessor()
                logger.info("Initialized Preprocessor for ZIP code lookup")
            
            # Look for 5-digit ZIP code pattern
            zip_matches = re.findall(r'\b(\d{5})\b', query)
            
            if not zip_matches:
                return None
            
            # Try each ZIP code found in the text using the preprocessor
            for zip_code in zip_matches:
                state = self._preprocessor.get_state_from_zip(zip_code)
                if state:
                    logger.info(f"Matched ZIP code {zip_code} to state {state} using preprocessor")
                    return state
                        
            # No valid ZIP code found, fall back to our original implementation
            return self._get_state_from_zip_fallback(query)
            
        except ImportError:
            logger.error("Could not import Preprocessor, falling back to simplified approach")
            return self._get_state_from_zip_fallback(query)
    
    def _get_state_from_zip_fallback(self, query: str) -> Optional[str]:
        """
        Fallback method using a simplified mapping of ZIP codes to states
        
        Args:
            query (str): The query text that might contain a ZIP code
            
        Returns:
            Optional[str]: State code or None
        """
        # Look for 5-digit ZIP code pattern
        zip_matches = re.findall(r'\b(\d{5})\b', query)
        
        if not zip_matches:
            return None
            
        # Simple mapping of ZIP code ranges to states
        zip_ranges = {
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
        
        for zip_code in zip_matches:
            zip_int = int(zip_code)
            
            for state, (lower, upper) in zip_ranges.items():
                if lower <= zip_int <= upper:
                    logger.info(f"Matched ZIP code {zip_code} to state {state} using fallback")
                    return state
        
        return None 

    def _get_supportive_resources_list(self, state_code: str) -> List[Dict[str, str]]:
        """
        Get a list of supportive resources for the specified state
        
        Args:
            state_code (str): Two-letter state code
            
        Returns:
            List[Dict[str, str]]: List of supportive resources
        """
        # Default national resources
        resources = [
            {
                "name": "Planned Parenthood",
                "url": "https://www.plannedparenthood.org/",
                "description": "Healthcare provider offering reproductive health services including abortion."
            },
            {
                "name": "National Abortion Federation Hotline",
                "url": "https://prochoice.org/patients/naf-hotline/",
                "phone": "1-800-772-9100",
                "description": "Offers referrals to providers and financial assistance."
            },
            {
                "name": "Abortion Finder",
                "url": "https://www.abortionfinder.org/",
                "description": "Search tool to find verified abortion providers."
            }
        ]
        
        # Additional state-specific resources could be added here
        # This is a placeholder for future enhancement
        
        return resources 

    def _get_state_from_conversation(self, conversation_history: List[Dict]) -> Optional[str]:
        """
        Extract state information from conversation history
        
        Args:
            conversation_history (List[Dict]): Previous conversation messages
            
        Returns:
            Optional[str]: State code if found, None otherwise
        """
        if not conversation_history:
            return None
            
        # Look through conversation history for state mentions
        for msg in reversed(conversation_history):
            if isinstance(msg, dict):
                # Check if this message has a state code directly
                if msg.get("state_code"):
                    return msg["state_code"]
                
                # Check if this message has state info in user_location
                if msg.get("user_location") and isinstance(msg["user_location"], dict):
                    state_from_loc = self._get_state_from_location(msg["user_location"])
                    if state_from_loc:
                        return state_from_loc
                
                # Check the content for state mentions
                if msg.get("content"):
                    state_found = self._get_state_from_query(query=msg["content"])
                    if state_found:
                        return state_found
                        
                    # Also check for ZIP codes in the message
                    state_from_zip = self._get_state_from_zip(query=msg["content"])
                    if state_from_zip:
                        return state_from_zip
        
        # No state found in conversation history
        return None
        
    def _get_state_from_location(self, user_location: Dict[str, str]) -> Optional[str]:
        """
        Extract state code from user location data
        
        Args:
            user_location (Dict[str, str]): User location data
            
        Returns:
            Optional[str]: State code if found, None otherwise
        """
        if not user_location:
            return None
            
        # Check for state code directly in the location data
        if user_location.get("state_code"):
            state_code = user_location["state_code"].upper()
            if state_code in self.STATE_NAMES:
                return state_code
                
        # Check for state name in the location data
        if user_location.get("state"):
            state_name = user_location["state"].lower()
            if state_name in self.STATE_NAMES_LOWER:
                return self.STATE_NAMES_LOWER[state_name]
                
        # Check for ZIP code in the location data
        if user_location.get("zip_code"):
            return self._get_state_from_zip(query=user_location["zip_code"])
            
        return None 