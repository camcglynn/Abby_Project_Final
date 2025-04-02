import logging
import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Set
import re
from openai import OpenAI

logger = logging.getLogger(__name__)

class UnifiedClassifier:
    """
    Unified classifier that determines the type and characteristics of user queries.
    
    This classifier analyzes user messages to determine:
    1. Primary query type (knowledge, emotional, policy)
    2. Confidence scores for each aspect
    3. Specific subtopics or sensitive content flags
    4. Multi-aspect query detection
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        """
        Initialize the unified classifier
        
        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            model_name (str): OpenAI model to use for classification
        """
        logger.info(f"Initializing UnifiedClassifier with model {model_name}")
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Set up OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model_name
        
        # Classification prompt template
        self.classification_prompt = """You are an expert classifier for a reproductive health chatbot. 
Analyze the following user message and classify it into one or more of these categories:

1. Knowledge: Factual questions about reproductive health, medical information, or general education.
2. Emotional: Messages seeking emotional support, expressing personal situations, or asking for comfort.
3. Policy: Questions about legal issues, regulations, or policies related to reproductive health.

For each category, provide a confidence score (0.0-1.0) reflecting how strongly the message fits that category.
Also identify any sensitive topics present in the message.

User message: {message}

Recent conversation history (if available):
{history}

Respond in JSON format with:
{{
  "primary_type": "knowledge|emotional|policy",
  "is_multi_aspect": true|false,
  "confidence_scores": {{
    "knowledge": 0.0-1.0,
    "emotional": 0.0-1.0,
    "policy": 0.0-1.0
  }},
  "topics": ["list", "of", "relevant", "topics"],
  "sensitive_content": ["list", "of", "sensitive", "topics"],
  "contains_location": true|false,
  "detected_locations": ["list", "of", "locations"],
  "query_complexity": "simple|medium|complex"
}}
"""
        
        # Initialize emotion detection patterns
        self.emotion_patterns = {
            "afraid": ["afraid", "scared", "terrified", "fear", "panic", "anxiety", "anxious"],
            "sad": ["sad", "depressed", "unhappy", "upset", "miserable", "grief", "heartbroken"],
            "angry": ["angry", "mad", "furious", "enraged", "upset", "frustrated", "annoyed"],
            "confused": ["confused", "unsure", "uncertain", "don't understand", "unclear", "lost"],
            "worried": ["worried", "concerned", "nervous", "stress", "distressed", "uneasy"],
            "hopeful": ["hopeful", "optimistic", "positive", "encouraged", "excited"],
            "grateful": ["grateful", "thankful", "appreciative", "relief", "relieved"]
        }
        
        # Maintain a cache of recent classifications to avoid redundant API calls
        self.classification_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def classify(self, message: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify a user message into appropriate categories
        
        Args:
            message (str): User message to classify
            conversation_history (List[Dict[str, Any]]): Previous conversation messages
            
        Returns:
            Dict[str, Any]: Classification results
        """
        try:
            # Format history for the prompt if available
            formatted_history = self._format_conversation_history(conversation_history)
            
            # Format the classification prompt
            prompt = self.classification_prompt.format(
                message=message,
                history=formatted_history
            )
            
            # Request classification from OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Extract and parse the response
            response_text = response.choices[0].message.content
            
            try:
                classification = json.loads(response_text)
                logger.info(f"Classification result: {classification}")
                
                # Validate classification has required fields
                if not all(k in classification for k in ["primary_type", "confidence_scores"]):
                    logger.warning("Missing required fields in classification response")
                    return self._create_default_classification(message)
                
                # Ensure all confidence scores are present
                if not all(k in classification["confidence_scores"] for k in ["knowledge", "emotional", "policy"]):
                    logger.warning("Missing confidence scores in classification response")
                    classification["confidence_scores"] = {
                        "knowledge": classification["confidence_scores"].get("knowledge", 0.33),
                        "emotional": classification["confidence_scores"].get("emotional", 0.33),
                        "policy": classification["confidence_scores"].get("policy", 0.33)
                    }
                
                return classification
                
            except json.JSONDecodeError:
                logger.error("Failed to parse classification response as JSON")
                return self._create_default_classification(message)
                
        except Exception as e:
            logger.error(f"Error classifying message: {str(e)}", exc_info=True)
            return self._create_default_classification(message)
    
    def _format_conversation_history(self, conversation_history: Optional[List[Dict[str, Any]]]) -> str:
        """
        Format conversation history for inclusion in the classification prompt
        
        Args:
            conversation_history (Optional[List[Dict[str, Any]]]): Previous conversation messages
            
        Returns:
            str: Formatted conversation history string
        """
        if not conversation_history:
            return "No conversation history available."
        
        # Limit to last 5 messages for context
        recent_history = conversation_history[-5:]
        
        formatted_history = []
        for msg in recent_history:
            role = msg.get("role", msg.get("sender", "unknown"))
            content = msg.get("content", msg.get("message", ""))
            formatted_history.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history)
    
    def _create_default_classification(self, message: str) -> Dict[str, Any]:
        """
        Create a default classification when the main classifier fails
        
        Args:
            message (str): The original message
            
        Returns:
            Dict[str, Any]: Default classification
        """
        logger.info("Creating default classification")
        
        return {
            "primary_type": "knowledge",
            "is_multi_aspect": False,
            "confidence_scores": {
                "knowledge": 0.7,
                "emotional": 0.2,
                "policy": 0.1
            },
            "topics": ["general", "reproductive_health"],
            "sensitive_content": [],
            "contains_location": False,
            "detected_locations": [],
            "query_complexity": "simple"
        }
    
    async def classify_message(self, message: str, 
                              history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Classify a message to identify multiple aspects with confidence scores
        
        Args:
            message (str): The user's message
            history (Optional[List[Dict[str, Any]]]): Conversation history
            
        Returns:
            Dict[str, Any]: Classification results with aspects and confidence scores
        """
        try:
            # First extract any location context with regex (more efficient)
            location_context = self._extract_location(message)
            
            # Check cache for identical recent queries
            cache_key = f"{message}_{str(location_context)}"
            if cache_key in self.classification_cache:
                logger.info(f"Using cached classification for query: {message[:50]}...")
                return self.classification_cache[cache_key]
            
            # Attempt GPT-4o classification first
            if self.client:
                try:
                    classification = await self._classify_with_gpt(message, history)
                    
                    # Add location context from regex extraction if not already present
                    if location_context and not classification.get("location_context"):
                        classification["location_context"] = location_context
                    
                    # Cache the classification
                    self.classification_cache[cache_key] = classification
                    return classification
                except Exception as e:
                    logger.error(f"Error with GPT classification: {str(e)}")
                    # Fall back to rule-based classification
            
            # Fallback to rule-based classification
            classification = self._rule_based_classification(message, history, location_context)
            
            # Cache the classification
            self.classification_cache[cache_key] = classification
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying message: {str(e)}", exc_info=True)
            # Return a simple fallback classification
            return {
                "aspects": [{"type": "knowledge", "confidence": 0.8}],
                "is_abortion_related": "abortion" in message.lower(),
                "location_context": location_context,
                "emotions": [],
                "overall_confidence": 0.5
            }
    
    async def _classify_with_gpt(self, message: str, 
                                history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Classify a message using GPT-4o with structured JSON output
        
        Args:
            message (str): The user's message
            history (Optional[List[Dict[str, Any]]]): Conversation history
            
        Returns:
            Dict[str, Any]: Classification results
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Define the prompt for GPT-4o classification
        system_message = """You are a specialized classifier for reproductive health questions.
Your task is to analyze the query and identify MULTIPLE aspects it might contain.
A single query might be asking for knowledge information, emotional support, and policy information at the same time.

Return a JSON object with the following fields:
- aspects: array of objects, each with "type" (one of: "knowledge", "emotional", "policy") and "confidence" (0.0-1.0)
- is_abortion_related: boolean indicating if any aspect relates to abortion
- location_context: string with location mentioned in query, or null if none
- emotions: array of emotion objects detected, each with "type" and "confidence"
- overall_confidence: number between 0 and 1 indicating overall confidence in the classification

Example response:
{
  "aspects": [
    {"type": "knowledge", "confidence": 0.9},
    {"type": "emotional", "confidence": 0.7},
    {"type": "policy", "confidence": 0.3}
  ],
  "is_abortion_related": true,
  "location_context": "Texas",
  "emotions": [
    {"type": "worried", "confidence": 0.8},
    {"type": "confused", "confidence": 0.4}
  ],
  "overall_confidence": 0.85
}

DO NOT include any text outside the JSON object. The JSON should be valid and parseable."""

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
            # First try with GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for better classification accuracy
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"{context}User message: {message}"}
                ],
                temperature=0.1,  # Low temperature for more consistent results
                max_tokens=500
            )
            
            logger.info("Successfully used GPT-4o for classification")
            
            # Extract and parse the JSON response
            result_text = response.choices[0].message.content.strip()
            
            # Safely parse JSON with fallbacks
            try:
                classification = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract the JSON part if there's text around it
                json_match = re.search(r'({.*})', result_text, re.DOTALL)
                if json_match:
                    try:
                        classification = json.loads(json_match.group(1))
                    except:
                        raise ValueError("Failed to parse classification JSON")
                else:
                    raise ValueError("Failed to parse classification JSON")
            
            # Ensure we have the expected structure
            if "aspects" not in classification:
                classification["aspects"] = [{"type": "knowledge", "confidence": 0.8}]
            
            # Log the classification result
            logger.info(f"GPT classified message: {classification}")
            return classification
            
        except Exception as e:
            logger.error(f"Error with GPT-4o: {str(e)}. Trying fallback to GPT-3.5-turbo")
            
            # Try fallback to GPT-3.5-turbo
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Fallback to GPT-3.5-turbo
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"{context}User message: {message}"}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                logger.info("Successfully used GPT-3.5-turbo for classification")
                
                # Extract and parse the JSON response with the same fallbacks
                result_text = response.choices[0].message.content.strip()
                
                try:
                    classification = json.loads(result_text)
                except json.JSONDecodeError:
                    json_match = re.search(r'({.*})', result_text, re.DOTALL)
                    if json_match:
                        try:
                            classification = json.loads(json_match.group(1))
                        except:
                            raise ValueError("Failed to parse classification JSON from GPT-3.5")
                    else:
                        raise ValueError("Failed to parse classification JSON from GPT-3.5")
                
                # Ensure we have the expected structure
                if "aspects" not in classification:
                    classification["aspects"] = [{"type": "knowledge", "confidence": 0.7}]
                
                logger.info(f"GPT-3.5 classified message: {classification}")
                return classification
                
            except Exception as fallback_error:
                logger.error(f"Error with GPT-3.5 fallback: {str(fallback_error)}")
                raise
    
    def _rule_based_classification(self, message: str, 
                                  history: Optional[List[Dict[str, Any]]] = None,
                                  location_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform rule-based classification as fallback
        
        Args:
            message (str): The user's message
            history (Optional[List[Dict[str, Any]]]): Conversation history
            location_context (Optional[str]): Pre-extracted location
        
        Returns:
            Dict[str, Any]: Classification results
        """
        message_lower = message.lower()
        
        # Initialize aspects list
        aspects = []
        
        # 1. Check for knowledge aspect (informational questions)
        knowledge_indicators = ["what", "how", "when", "where", "why", "who", "which", 
                               "can you", "tell me", "explain", "information", "learn", "know"]
        
        if any(indicator in message_lower for indicator in knowledge_indicators):
            aspects.append({"type": "knowledge", "confidence": 0.8})
        
        # 2. Check for emotional aspect
        emotion_words = []
        for emotion, patterns in self.emotion_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                emotion_words.append({"type": emotion, "confidence": 0.7})
                if {"type": "emotional", "confidence": 0.7} not in aspects:
                    aspects.append({"type": "emotional", "confidence": 0.7})
        
        # 3. Check for policy aspect
        policy_indicators = ["law", "legal", "illegal", "allowed", "banned", "policy", "right", "access",
                            "restrictions", "permit", "permission", "mandate", "require", "regulation"]
        
        is_policy = any(indicator in message_lower for indicator in policy_indicators)
        has_location = bool(location_context)
        
        # Higher confidence if both policy indicators and location are present
        if is_policy and has_location:
            aspects.append({"type": "policy", "confidence": 0.9})
        elif is_policy:
            aspects.append({"type": "policy", "confidence": 0.7})
        elif has_location and "abortion" in message_lower:
            aspects.append({"type": "policy", "confidence": 0.8})
        
        # 4. Ensure we have at least one aspect
        if not aspects:
            # Default to knowledge with medium confidence
            aspects.append({"type": "knowledge", "confidence": 0.6})
        
        # 5. Check if question is abortion-related
        abortion_terms = ["abortion", "terminate", "termination", "pregnancy termination", 
                         "roe v wade", "pro-choice", "pro-life", "end pregnancy"]
        is_abortion_related = any(term in message_lower for term in abortion_terms)
        
        # 6. Build final classification
        classification = {
            "aspects": aspects,
            "is_abortion_related": is_abortion_related,
            "location_context": location_context,
            "emotions": emotion_words,
            "overall_confidence": 0.7
        }
        
        logger.info(f"Rule-based classified message: {classification}")
        return classification
    
    def _extract_location(self, message: str) -> Optional[str]:
        """
        Extract location mentions from the message
        
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
        
        # State abbreviations mapping
        state_abbrs = {v.lower(): v for k, v in us_states.items()}
        
        # Look for state names in the message
        message_lower = message.lower()
        words = message_lower.split()
        
        # First try direct state name matches
        for state_name, code in us_states.items():
            if state_name in message_lower:
                return code
        
        # Then try abbreviations
        for abbr_lower, code in state_abbrs.items():
            if abbr_lower in words:
                return code
        
        # Look for "in [state]" pattern
        in_state_match = re.search(r'in\s+([a-zA-Z\s]+?)(?:\s|$|\.|\?|,)', message_lower)
        if in_state_match:
            potential_state = in_state_match.group(1).strip()
            # Check if this is a state name
            for state_name, code in us_states.items():
                if potential_state in state_name or state_name in potential_state:
                    return code
        
        # No location found
        return None 