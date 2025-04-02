import logging
import os
import time
from typing import Dict, List, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class EmotionalSupportHandler:
    """
    Handler for emotional support aspects of user queries.
    
    This class processes messages seeking emotional support,
    validating user emotions and providing compassionate responses.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        """
        Initialize the emotional support handler
        
        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            model_name (str): OpenAI model to use
        """
        logger.info(f"Initializing EmotionalSupportHandler with model {model_name}")
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Set up OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model_name
        
        # Common emotions in reproductive health contexts
        self.emotion_categories = {
            "anxiety": ["anxious", "nervous", "worried", "scared", "fearful", "afraid", "panic", "stressed"],
            "grief": ["grief", "loss", "sad", "mourning", "devastated", "heartbroken", "depressed"],
            "confusion": ["confused", "uncertain", "unsure", "undecided", "torn", "conflicted", "overwhelmed"],
            "shame": ["ashamed", "embarrassed", "humiliated", "guilt", "regret", "blame"],
            "hope": ["hopeful", "optimistic", "excited", "looking forward", "positive"],
            "anger": ["angry", "frustrated", "mad", "annoyed", "upset", "resentful", "bitter"]
        }
        
        # Support response prompt template
        self.support_prompt = """You are a compassionate and supportive reproductive health counselor.

User message: {query}

Full message context: {full_message}

Detected emotions: {detected_emotions}

Conversation history: {conversation_history}

Provide empathetic and supportive guidance to the user. Validate their emotions without judgment.
Remember these guidelines:
1. Show empathy and validate feelings
2. Maintain a non-judgmental tone
3. Offer support while respecting user autonomy
4. Avoid minimizing their concerns
5. Balance emotional support with factual information if appropriate
6. If needed, gently suggest professional resources

Focus on emotional support first. Keep your response compassionate and person-centered.
"""
    
    async def process_query(self, 
                          query: str, 
                          full_message: str = None,
                          conversation_history: List[Dict[str, Any]] = None,
                          user_location: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process an emotional support query
        
        Args:
            query (str): The specific emotional support query aspect
            full_message (str): The original complete user message
            conversation_history (List[Dict[str, Any]]): Previous conversation messages
            user_location (Optional[Dict[str, str]]): User's location data (not used by emotional handler)
            
        Returns:
            Dict[str, Any]: Response with supportive text
        """
        try:
            start_time = time.time()
            logger.info(f"Processing emotional support query: {query[:100]}...")
            
            # Use the provided query or full message if query is None
            query_text = query or full_message
            if not query_text:
                raise ValueError("No query text provided")
            
            # Detect emotions in the query
            detected_emotions = self._detect_emotions(query_text)
            
            # Format conversation history
            formatted_history = self._format_conversation_history(conversation_history)
            
            # Generate supportive response using OpenAI
            prompt = self.support_prompt.format(
                query=query_text,
                full_message=full_message or query_text,
                detected_emotions=", ".join(detected_emotions) if detected_emotions else "None specifically detected",
                conversation_history=formatted_history
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7  # Higher temperature for more diverse emotional responses
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Log processing time
            processing_time = time.time() - start_time
            logger.info(f"Emotional support response generated in {processing_time:.2f} seconds")
            
            return {
                "text": response_text,
                "detected_emotions": detected_emotions,
                "citations": [],  # Emotional support typically doesn't include citations
                "citation_objects": [],
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing emotional support query: {str(e)}", exc_info=True)
            return {
                "text": "I understand you might be going through a difficult time. While I'm here to support you, I'm having trouble understanding the specifics of what you're feeling. Could you share a bit more about what's on your mind?",
                "detected_emotions": [],
                "citations": [],
                "citation_objects": []
            }
    
    def _detect_emotions(self, text: str) -> List[str]:
        """
        Detect emotions expressed in the text
        
        Args:
            text (str): The text to analyze
            
        Returns:
            List[str]: List of detected emotion categories
        """
        text_lower = text.lower()
        detected = set()
        
        # Simple keyword-based emotion detection
        for category, keywords in self.emotion_categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected.add(category)
                    break
        
        return list(detected)
    
    def _format_conversation_history(self, conversation_history: Optional[List[Dict[str, Any]]]) -> str:
        """
        Format conversation history for inclusion in the support prompt
        
        Args:
            conversation_history (Optional[List[Dict[str, Any]]]): Previous conversation messages
            
        Returns:
            str: Formatted conversation history string
        """
        if not conversation_history:
            return "No previous conversation."
        
        # Limit to last 3 messages for context
        recent_history = conversation_history[-3:]
        
        formatted_history = []
        for msg in recent_history:
            role = msg.get("role", msg.get("sender", "unknown"))
            content = msg.get("content", msg.get("message", ""))
            formatted_history.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history) 