import logging
import json
import os
from typing import Dict, List, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class AspectDecomposer:
    """
    Decomposes complex user queries into multiple aspects for specialized handling.
    
    This class analyzes user messages and breaks them down into:
    1. Knowledge-seeking components
    2. Emotional support needs
    3. Policy and legal information requests
    
    Each aspect is processed separately by a specialized handler.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        """
        Initialize the aspect decomposer
        
        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            model_name (str): OpenAI model to use for decomposition
        """
        logger.info(f"Initializing AspectDecomposer with model {model_name}")
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Set up OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model_name
        
        # Decomposition prompt template
        self.decomposition_prompt = """You are an expert query analyzer for a reproductive health chatbot. 
Your task is to analyze the user's message and break it down into different aspects for specialized processing.

User message: {message}

Classification information: {classification}

For complex or multi-faceted queries, decompose the message into separate aspects:
1. Knowledge: Factual or medical information needs
2. Emotional: Emotional support or personal situation aspects
3. Policy: Legal information or policy-related questions

IMPORTANT: If the message contains multiple distinct questions or needs, identify EACH ONE as a separate aspect.
For example, if a user asks:
- "What are abortion laws in Texas and I'm feeling anxious about being pregnant"
- This should be split into a POLICY aspect (abortion laws) and an EMOTIONAL aspect (anxiety)

Similarly, if they ask:
- "Can you tell me about IUDs and what states offer free birth control?"
- This should be split into a KNOWLEDGE aspect (about IUDs) and a POLICY aspect (state coverage)

For simple queries with only one question type, create just one aspect of the appropriate type.

Recent conversation history (if available):
{history}

Respond in JSON format with:
{{
  "aspects": [
    {{
      "type": "knowledge|emotional|policy",
      "query": "rephrased aspect-specific query that focuses solely on this aspect",
      "confidence": 0.0-1.0,
      "topics": ["list", "of", "relevant", "topics"],
      "requires_state_context": true|false
    }},
    // Additional aspects if needed
  ]
}}

NOTE: For policy-related questions that involve state laws or regulations, set "requires_state_context" to true.
"""
    
    async def decompose(self, 
                       message: str, 
                       classification: Dict[str, Any],
                       conversation_history: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Decompose a user message into multiple aspects based on classification
        
        Args:
            message (str): User message to decompose
            classification (Dict[str, Any]): Classification results
            conversation_history (List[Dict[str, Any]]): Previous conversation messages
            
        Returns:
            List[Dict[str, Any]]: List of aspect dictionaries
        """
        try:
            # Quick check if decomposition is necessary
            # If single aspect with high confidence, we can skip complex decomposition
            confidence_scores = classification.get("confidence_scores", {})
            is_multi_aspect = classification.get("is_multi_aspect", False)
            primary_type = classification.get("primary_type", "knowledge")
            
            # If it's a simple query, return a single aspect and skip API call
            if not is_multi_aspect and confidence_scores.get(primary_type, 0) > 0.8:
                logger.info(f"Simple {primary_type} query detected, skipping complex decomposition")
                return [{
                    "type": primary_type,
                    "query": message,
                    "confidence": confidence_scores.get(primary_type, 0.9),
                    "topics": classification.get("topics", ["reproductive_health"])
                }]
            
            # Format history for the prompt if available
            formatted_history = self._format_conversation_history(conversation_history)
            
            # Format the decomposition prompt
            prompt = self.decomposition_prompt.format(
                message=message,
                classification=json.dumps(classification),
                history=formatted_history
            )
            
            # Request decomposition from OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Extract and parse the response
            response_text = response.choices[0].message.content
            
            try:
                decomposition = json.loads(response_text)
                aspects = decomposition.get("aspects", [])
                logger.info(f"Decomposed into {len(aspects)} aspects")
                
                # Validate response format
                if not aspects:
                    return self._create_default_decomposition(message, classification)
                
                # Ensure all required fields are present in each aspect
                valid_aspects = []
                for aspect in aspects:
                    if "type" in aspect and "query" in aspect:
                        # Add default confidence if missing
                        if "confidence" not in aspect:
                            aspect["confidence"] = confidence_scores.get(aspect["type"], 0.7)
                            
                        valid_aspects.append(aspect)
                
                return valid_aspects if valid_aspects else self._create_default_decomposition(message, classification)
                
            except json.JSONDecodeError:
                logger.error("Failed to parse decomposition response as JSON")
                return self._create_default_decomposition(message, classification)
                
        except Exception as e:
            logger.error(f"Error decomposing message: {str(e)}", exc_info=True)
            return self._create_default_decomposition(message, classification)
    
    def _format_conversation_history(self, conversation_history: Optional[List[Dict[str, Any]]]) -> str:
        """
        Format conversation history for inclusion in the decomposition prompt
        
        Args:
            conversation_history (Optional[List[Dict[str, Any]]]): Previous conversation messages
            
        Returns:
            str: Formatted conversation history string
        """
        if not conversation_history:
            return "No conversation history available."
        
        # Limit to last 3 messages for context (to keep prompt shorter)
        recent_history = conversation_history[-3:]
        
        formatted_history = []
        for msg in recent_history:
            role = msg.get("role", msg.get("sender", "unknown"))
            content = msg.get("content", msg.get("message", ""))
            formatted_history.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history)
    
    def _create_default_decomposition(self, message: str, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create a default decomposition when the main decomposer fails
        
        Args:
            message (str): The original message
            classification (Dict[str, Any]): The message classification
            
        Returns:
            List[Dict[str, Any]]: Default aspect list
        """
        logger.info("Creating default decomposition")
        
        primary_type = classification.get("primary_type", "knowledge")
        confidence = classification.get("confidence_scores", {}).get(primary_type, 0.7)
        topics = classification.get("topics", ["reproductive_health"])
        
        return [{
            "type": primary_type,
            "query": message,
            "confidence": confidence,
            "topics": topics
        }] 