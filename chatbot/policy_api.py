import os
import logging
import requests
import json
import re
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class PolicyAPI:
    """
    Integration with the abortion policy API to provide up-to-date policy information
    """

    def __init__(self):
        """Initialize the Policy API"""
        # Default API key (will be overridden by environment variable if available)
        default_api_key = ''
        
        # Try to get API key from environment vars
        self.api_key = os.environ.get('ABORTION_POLICY_API_KEY', default_api_key)
        self.base_url = "https://api.abortionpolicyapi.com/v1"
        
        if self.api_key:
            logger.info("Abortion Policy API key found")
        else:
            logger.warning("No Abortion Policy API key found in environment")
        
        # Define US state codes and names
        self.STATE_NAMES = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
        }
        
        # Create lowercase state names for searches
        self.STATE_NAMES_LOWER = {k.lower(): k for k in self.STATE_NAMES.keys()}
        for k, v in self.STATE_NAMES.items():
            self.STATE_NAMES_LOWER[v.lower()] = k
        
        # We'll lazy-load GPT if needed
        self.gpt_model = None
        logger.info("Policy API initialized successfully")

    def _extract_state_from_question(self, question: str) -> Optional[str]:
        """
        Extract state information from a user's question. 
        Return the 2-letter state code if found, else None.
        """
        question_lower = question.lower().strip()
        
        # List of common words that are also state abbreviations
        ambiguous_abbrs = {
            'in': 'Indiana',
            'me': 'Maine',
            'or': 'Oregon',
            'hi': 'Hawaii',
            'id': 'Idaho',
            'la': 'Louisiana',
            'ma': 'Massachusetts',
            'md': 'Maryland',
            'mo': 'Missouri',
            'oh': 'Ohio',
            'ok': 'Oklahoma',
            'pa': 'Pennsylvania',
            'wa': 'Washington'
        }
        
        # Check for state indicators first
        state_indicators = ['in', 'at', 'for', 'about', 'state of', 'policy in', 'laws in']
        has_state_indicator = any(indicator + ' ' in question_lower + ' ' for indicator in state_indicators)
        
        # Quick pass: check exact name matches (e.g. "california")
        for state_name_lower, code in self.STATE_NAMES_LOWER.items():
            if state_name_lower in question_lower:
                # Only return if there's a state indicator or it's a single-state query
                words = question_lower.split()
                if has_state_indicator or (len(words) <= 3 and state_name_lower in words):
                    logger.debug(
                        f"Found state name '{state_name_lower}' in question -> code={code}"
                    )
                    return code

        # Check for known 2-letter abbreviations in the text
        # (Only valid if they match an actual state code)
        pattern = r'\b([A-Za-z]{2})\b'
        matches = re.findall(pattern, question)
        for match in matches:
            abbr = match.upper()
            if abbr in self.STATE_NAMES:
                # For ambiguous abbreviations, require more context
                if match.lower() in ambiguous_abbrs:
                    # Check if it's preceded by specific location phrases
                    location_phrases = ['in', 'state of', 'from', 'to', 'living in', 'located in', 'moving to']
                    has_location_phrase = any(phrase + ' ' + match.lower() in question_lower for phrase in location_phrases)
                    if not has_location_phrase:
                        continue
                
                # Only return if there's a state indicator or it's a single-state query
                if has_state_indicator or len(question_lower.split()) <= 3:
                    logger.debug(f"Found state abbreviation '{abbr}' in question")
                    return abbr

        # If nothing found, return None
        logger.debug("No valid US state recognized in the question")
        return None

    def get_policy_data(self, state_code):
        """Get abortion policy data for a state"""
        try:
            # Normalize state code
            state_code = state_code.upper()
            state_name = self.STATE_NAMES.get(state_code, state_code)
            
            # Define endpoints to query
            endpoints = {
                "waiting_periods": "waiting_periods",
                "insurance_coverage": "insurance_coverage",
                "gestational_limits": "gestational_limits",
                "minors": "minors"
            }
            
            # Use correct API version and headers format
            base_url = "https://api.abortionpolicyapi.com/v1"
            headers = {'token': self.api_key}
            
            # Collect data from all endpoints
            policy_info = {}
            logger.info(f"Requesting policy data for {state_name} from API")
            
            for key, endpoint in endpoints.items():
                url = f"{base_url}/{endpoint}/states/{state_code}"
                time.sleep(0.5)  # Avoid rate limiting
                logger.info(f"Querying endpoint: {url}")
                
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if data:  # if data is not empty
                        policy_info[key] = data
                        logger.info(f"Successfully received data for {key}")
                else:
                    logger.error(f"Error getting {key} for {state_code}: {response.status_code}")
                    policy_info[key] = {}
            
            # Format the results
            if policy_info:
                logger.info(f"Successfully received policy data for {state_name}")
                return self._format_api_response(policy_info, state_name)
            else:
                logger.error(f"No policy data found for {state_code}")
                return {
                    'error': True,
                    'legal_status': f"Unable to retrieve policy information for {state_name}",
                    'gestational_limit': "Information currently unavailable",
                    'restrictions': ["Please contact Planned Parenthood or National Abortion Federation for accurate information"],
                    'services': ["Contact Planned Parenthood or check ineedana.com for services in your area"],
                    'resources': [
                        "Planned Parenthood",
                        "National Abortion Federation",
                        "INeedAnA.com"
                    ],
                    'sources': [
                        {
                            'title': 'Planned Parenthood',
                            'url': 'https://www.plannedparenthood.org/',
                            'accessed_date': datetime.now().strftime('%Y-%m-%d')
                        }
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error in get_policy_data for {state_code}: {str(e)}")
            return {
                'error': True,
                'legal_status': f"An error occurred while retrieving policy information for {state_name}",
                'gestational_limit': "Information currently unavailable",
                'restrictions': ["Please contact Planned Parenthood or National Abortion Federation for accurate information"],
                'services': ["Contact Planned Parenthood or check ineedana.com for services in your area"],
                'resources': [
                    "Planned Parenthood",
                    "National Abortion Federation",
                    "INeedAnA.com"
                ],
                'sources': [
                    {
                        'title': 'Planned Parenthood',
                        'url': 'https://www.plannedparenthood.org/',
                        'accessed_date': datetime.now().strftime('%Y-%m-%d')
                    }
                ]
            }

    def _format_api_response(self, api_data, state_name):
        """Format raw API response into structured policy data"""
        try:
            # Extract and organize information from multiple endpoints
            legal_status = "Information based on current state restrictions"
            gestational_limit = "Varies by state law"
            restrictions = []
            services = []
            resources = [
                f"Planned Parenthood in {state_name}",
                "National Abortion Federation Hotline: 1-800-772-9100",
                "INeedAnA.com"
            ]
            
            # Extract waiting periods info
            if 'waiting_periods' in api_data and api_data['waiting_periods']:
                for state, info in api_data['waiting_periods'].items():
                    if info.get('waiting_period_hours'):
                        restrictions.append(f"Waiting period: {info.get('waiting_period_hours')} hours")
                    if info.get('counseling_visits'):
                        restrictions.append(f"Required counseling visits: {info.get('counseling_visits')}")
            
            # Extract gestational limits
            if 'gestational_limits' in api_data and api_data['gestational_limits']:
                for state, info in api_data['gestational_limits'].items():
                    if info.get('banned_after_weeks_since_LMP'):
                        gestational_limit = f"Banned after {info.get('banned_after_weeks_since_LMP')} weeks since last menstrual period"
                    if info.get('exception_life'):
                        restrictions.append("Exception for life endangerment: Yes")
                    else:
                        restrictions.append("Exception for life endangerment: No")
                    if info.get('exception_health'):
                        restrictions.append("Exception for health: Yes")
                    else:
                        restrictions.append("Exception for health: No")
            
            # Extract insurance coverage
            if 'insurance_coverage' in api_data and api_data['insurance_coverage']:
                for state, info in api_data['insurance_coverage'].items():
                    if info.get('private_coverage_no_restrictions'):
                        services.append("Private insurance can cover abortion without restrictions")
                    else:
                        restrictions.append("Restrictions on private insurance coverage")
                    if info.get('exchange_coverage'):
                        services.append("Exchange insurance coverage available")
                    if info.get('medicaid_coverage'):
                        services.append("Medicaid coverage available")
            
            # Extract minors info
            if 'minors' in api_data and api_data['minors']:
                for state, info in api_data['minors'].items():
                    if info.get('parental_consent_required'):
                        restrictions.append("Parental consent required for minors")
                    if info.get('parental_notification_required'):
                        restrictions.append("Parental notification required for minors")
                    if info.get('judicial_bypass_available'):
                        services.append("Judicial bypass available for minors")
            
            return {
                'legal_status': legal_status,
                'gestational_limit': gestational_limit,
                'restrictions': restrictions,
                'services': services,
                'resources': resources,
                'sources': [
                    {
                        'title': 'Abortion Policy API',
                        'url': 'https://www.abortionpolicyapi.com/',
                        'accessed_date': datetime.now().strftime('%Y-%m-%d')
                    },
                    {
                        'title': 'Planned Parenthood',
                        'url': 'https://www.plannedparenthood.org/learn/abortion',
                        'accessed_date': datetime.now().strftime('%Y-%m-%d')
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error formatting API response: {str(e)}")
            return None

    def _get_fallback_data(self, state_code):
        """Generate fallback policy data for a state when API fails"""
        state_name = self.STATE_NAMES.get(state_code, state_code)
        logger.info(f"Using fallback data for {state_name} ({state_code})")
        
        # Default data structure with placeholder text
        return {
            'legal_status': f"Information about legal status in {state_name} is currently unavailable",
            'gestational_limit': "Information about gestational limits is currently unavailable",
            'restrictions': [
                "For the most up-to-date information on restrictions in your state, please consult local resources",
                "State laws and regulations can change frequently"
            ],
            'services': [
                "For information about available services, contact Planned Parenthood",
                "The National Abortion Federation hotline: 1-800-772-9100",
                "INeedAnA.com can provide location-specific information"
            ],
            'resources': [
                "Planned Parenthood",
                "National Abortion Federation",
                "INeedAnA.com",
                "Local healthcare providers"
            ],
            'sources': [
                {
                    'title': 'Planned Parenthood',
                    'url': 'https://www.plannedparenthood.org/',
                    'accessed_date': datetime.now().strftime('%Y-%m-%d')
                },
                {
                    'title': 'National Abortion Federation',
                    'url': 'https://prochoice.org/patients/find-a-provider/',
                    'accessed_date': datetime.now().strftime('%Y-%m-%d')
                }
            ]
        }

    def get_policy_response(self,
                            question: str,
                            conversation_history: Optional[List[Dict[str, Any]]] = None,
                            location_context: Optional[str] = None) -> str:
        """
        Process a policy-related question and return a response:
         - If no valid US state is found, we disclaim that we only have US data.
         - Otherwise, fetch data from the API and format the response.
        """
        # 1) Check location_context first
        state_code = None
        if location_context:
            # If location_context is recognized in our state map
            if location_context.lower() in self.STATE_NAMES_LOWER:
                state_code = self.STATE_NAMES_LOWER[location_context.lower()]
                logger.info(
                    f"Converted location context '{location_context}' -> state code '{state_code}'"
                )

        # 2) If still None, try extracting from question
        if not state_code:
            extracted_code = self._extract_state_from_question(question)
            if extracted_code:
                state_code = extracted_code
                logger.info(
                    f"Extracted state code from question: {state_code}")

        # 3) If still None, check conversation history for state context
        if not state_code and conversation_history:
            for msg in reversed(conversation_history[-3:]):  # Check last 3 messages
                if msg['sender'] == 'user':
                    extracted_code = self._extract_state_from_question(msg['message'])
                    if extracted_code:
                        state_code = extracted_code
                        logger.info(f"Found state code in conversation history: {state_code}")
                        break

        # 4) If still None, we need to ask for state information
        if not state_code:
            logger.info("No state context found, requesting state information")
            return (
                "I understand you're looking for information about abortion access. To provide you with accurate "
                "information about the laws and regulations in your area, could you please let me know which state "
                "you're in? Different states have different policies, and I want to make sure I give you the most "
                "relevant information."
            )

        # 5) Fetch policy data
        policy_data = self.get_policy_data(state_code)
        if "error" in policy_data:
            logger.warning(
                f"Error in policy data for state '{state_code}': {policy_data['error']}"
            )
            return (
                f"I'm sorry, I'm having trouble accessing policy information for {state_code}. "
                "Abortion policies vary by state and may change. "
                "You might consider contacting Planned Parenthood or a local provider for the most current details."
            )

        # 6) Format policy data into a user-friendly response
        return self._format_policy_response(question, state_code, policy_data)

    def _format_policy_response(self, question: str, state_code: str,
                                policy_data: Dict[str, Any]) -> str:
        """
        Format policy data into a user-friendly response using only data from the API.
        No GPT-generated content is used.
        """
        state_name = policy_data["state_name"]
        
        # Determine state restrictiveness to provide appropriate empathetic response
        restrictiveness_level = self._get_state_restrictiveness(state_code, policy_data)
        
        # Prepare empathetic prefix based on restrictiveness level
        empathetic_prefix = ""
        if restrictiveness_level == "restrictive":
            empathetic_prefix = (
                f"I understand this may be difficult to hear, but {state_name} has restrictive abortion laws. "
                f"I know this can be stressful and concerning if you're seeking care. "
                f"Here's what you should know: "
            )
        elif restrictiveness_level == "moderate":
            empathetic_prefix = (
                f"I understand you may be concerned about access to care in {state_name}. "
                f"While there are some restrictions in place, there are still options available. "
                f"Here's what you should know: "
            )
        elif restrictiveness_level == "supportive":
            empathetic_prefix = (
                f"I want to let you know that {state_name} generally has supportive policies for reproductive healthcare access. "
                f"This means you likely have several options available to you. "
                f"Here's what you should know: "
            )

        # Build response using only API data
        response_parts = []
        
        # Add legal status
        if "gestational_limits" in policy_data["endpoints"]:
            gestational_data = policy_data["endpoints"]["gestational_limits"]
            if isinstance(gestational_data, dict):
                if gestational_data.get("banned", False):
                    response_parts.append(f"Abortion is not available in {state_name}.")
                else:
                    response_parts.append(f"Abortion is available in {state_name}.")
                    
                    # Add gestational limit information
                    if "banned_after_weeks_since_LMP" in gestational_data:
                        limit = gestational_data["banned_after_weeks_since_LMP"]
                        if limit == 99 or limit == "no specific limit":
                            response_parts.append("There is no specific gestational limit.")
                        elif isinstance(limit, (int, float)):
                            response_parts.append(f"There is a gestational limit of {limit} weeks.")

        # Add insurance coverage information
        if "insurance_coverage" in policy_data["endpoints"]:
            insurance = policy_data["endpoints"]["insurance_coverage"]
            if isinstance(insurance, dict):
                if insurance.get("private_coverage_prohibited", False):
                    response_parts.append("Private insurance coverage for abortion is prohibited.")
                if insurance.get("exchange_coverage_prohibited", False):
                    response_parts.append("Exchange coverage for abortion is prohibited.")
                if insurance.get("medicaid_coverage_provider", "") == "yes":
                    response_parts.append("Medicaid coverage for abortion is available.")

        # Add waiting period information
        if "waiting_periods" in policy_data["endpoints"]:
            waiting = policy_data["endpoints"]["waiting_periods"]
            if isinstance(waiting, dict) and waiting.get("hours", 0) > 0:
                response_parts.append(f"There is a {waiting['hours']}-hour waiting period required.")

        # Add minors' rights information
        if "minors" in policy_data["endpoints"]:
            minors = policy_data["endpoints"]["minors"]
            if isinstance(minors, dict):
                if minors.get("parental_consent_required", False):
                    response_parts.append("Parental consent is required for minors.")
                elif minors.get("parental_notification_required", False):
                    response_parts.append("Parental notification is required for minors.")

        # Combine all parts into a response
        response = " ".join(response_parts)
        
        # Add empathetic prefix if available
        if empathetic_prefix:
            response = f"{empathetic_prefix}{response}"

        # Add source attribution
        response += " (Source: Abortion Policy API)"

        return response

    def _get_state_restrictiveness(self, state_code: str, policy_data: Dict[str, Any]) -> str:
        """
        Determine the restrictiveness level of a state's abortion policies.
        
        Args:
            state_code (str): The two-letter state code
            policy_data (Dict[str, Any]): Policy data from the API
            
        Returns:
            str: Restrictiveness level ('restrictive', 'moderate', 'supportive', or 'unknown')
        """
        try:
            # Classification is based on gestational limits and other restrictions
            
            # Default to unknown if we lack data
            if not policy_data or "endpoints" not in policy_data:
                return "unknown"
                
            # Check for banned states
            if "gestational_limits" in policy_data["endpoints"]:
                gestational_data = policy_data["endpoints"]["gestational_limits"]
                
                # Completely banned states (or effectively banned with very early limits)
                if isinstance(gestational_data, dict):
                    if "banned" in gestational_data and gestational_data.get("banned", False):
                        return "restrictive"
                    # States with 6-week bans (or earlier) are effectively highly restrictive
                    if "banned_after_weeks_since_LMP" in gestational_data:
                        limit = gestational_data["banned_after_weeks_since_LMP"]
                        if isinstance(limit, (int, float)) and limit <= 6:
                            return "restrictive"
                        # States with limits between 7-15 weeks are moderately restrictive
                        elif isinstance(limit, (int, float)) and limit <= 15:
                            return "moderate"
            
            # If we get here and haven't returned, check other indicators
            restrictive_indicators = 0
            supportive_indicators = 0
            
            # Check waiting periods
            if "waiting_periods" in policy_data["endpoints"]:
                waiting = policy_data["endpoints"]["waiting_periods"]
                if isinstance(waiting, dict) and waiting.get("hours", 0) >= 24:
                    restrictive_indicators += 1
            
            # Check for insurance coverage
            if "insurance_coverage" in policy_data["endpoints"]:
                insurance = policy_data["endpoints"]["insurance_coverage"]
                if isinstance(insurance, dict):
                    if insurance.get("private_coverage_prohibited", False):
                        restrictive_indicators += 1
                    if insurance.get("exchange_coverage_prohibited", False):
                        restrictive_indicators += 1
                    if insurance.get("medicaid_coverage_provider", "") == "yes":
                        supportive_indicators += 1
            
            # Check for parental involvement
            if "minors" in policy_data["endpoints"]:
                minors = policy_data["endpoints"]["minors"]
                if isinstance(minors, dict) and minors.get("parental_consent_required", False):
                    restrictive_indicators += 1
            
            # Determine level based on indicators
            if restrictive_indicators >= 2:
                return "restrictive"
            elif restrictive_indicators > supportive_indicators:
                return "moderate"
            elif supportive_indicators > 0:
                return "supportive"
            
            # If we still can't determine, check gestational limits again for later limits
            if "gestational_limits" in policy_data["endpoints"]:
                gestational_data = policy_data["endpoints"]["gestational_limits"]
                if isinstance(gestational_data, dict) and "banned_after_weeks_since_LMP" in gestational_data:
                    limit = gestational_data["banned_after_weeks_since_LMP"]
                    # If limit is 20+ weeks or no specific limit, state is likely supportive
                    if limit == "no specific limit" or limit == 99 or (isinstance(limit, (int, float)) and limit >= 20):
                        return "supportive"
            
            # If we can't determine clearly, default to moderate
            return "moderate"
            
        except Exception as e:
            logger.error(f"Error determining state restrictiveness: {str(e)}", exc_info=True)
            return "unknown"
    
    def get_state_code(self, location_context: str) -> Optional[str]:
        """Converts a location string (state name or abbreviation) into a 2-letter state code."""
        location_context = location_context.lower()
        if location_context in self.STATE_NAMES_LOWER:
            return self.STATE_NAMES_LOWER[location_context]
        
        # Check for known 2-letter abbreviations in the text
        # (Only valid if they match an actual state code)
        pattern = r'\b([A-Za-z]{2})\b'
        matches = re.findall(pattern, location_context)
        for match in matches:
            abbr = match.upper()
            if abbr in self.STATE_NAMES:
                return abbr
        return None

    def get_abortion_policy(self, location_context: str) -> str:
        """
        Get abortion policy information for a given location

        Args:
            location_context (str): The location to get policy for (state name or abbreviation)

        Returns:
            str: Policy information for the location
        """
        try:
            # Check if this is a non-US country
            if location_context.lower() in ['india', 'canada', 'uk', 'australia', 'mexico', 'france', 'germany', 
                                         'china', 'japan', 'brazil', 'spain', 'italy', 'russia', 'north korea']:
                logger.debug(f"Non-US country detected: {location_context}")
                # Return response WITHOUT Abortion Policy API citation marker
                # This ensures the citation manager won't add the wrong source
                return (
                    f"I'm sorry, but I don't have specific information about abortion access in {location_context.title()}. "
                    f"It's important to note that different countries have different regulations regarding reproductive healthcare. "
                    f"For accurate, up-to-date information, I recommend consulting local healthcare providers or international "
                    f"organizations like the World Health Organization (WHO) or the United Nations Population Fund that specialize in "
                    f"reproductive rights. Remember, prioritizing your safety and well-being is crucial.")

            # Convert to state code for API call
            state_code = self.get_state_code(location_context)

            # If no valid US state is found, we disclaim we only have US data
            if not state_code:
                logger.debug("No valid US state recognized -> returning fallback message.")
                return (
                    "I'm sorry, but I only have policy information for U.S. states right now. "
                    "I don't have data about abortion policies in that location.")

            # Fetch policy data
            policy_data = self.get_policy_data(state_code)
            if "error" in policy_data:
                logger.warning(
                    f"Error in policy data for state '{state_code}': {policy_data['error']}"
                )
                return (
                    f"I'm sorry, I'm having trouble accessing policy information for {state_code}. "
                    "Abortion policies vary by state and may change. "
                    "You might consider contacting Planned Parenthood or a local provider for the most current details."
                )

            # Format policy data into a user-friendly response
            return self._format_policy_response(location_context, state_code, policy_data)


        except Exception as e:
            logger.error(f"Error getting abortion policy: {e}", exc_info=True)
            return "I'm sorry, something went wrong. Please try again later."