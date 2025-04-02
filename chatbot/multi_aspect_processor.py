import logging
import asyncio
from typing import Dict, List, Any, Optional
import os
import time
import uuid

from .unified_classifier import UnifiedClassifier
from .aspect_decomposer import AspectDecomposer
from .response_composer import ResponseComposer
from .knowledge_handler import KnowledgeHandler
from .emotional_support_handler import EmotionalSupportHandler
from .policy_handler import PolicyHandler
from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)

class MultiAspectQueryProcessor:
    """
    Main orchestrator for the multi-aspect query handling process.
    
    This class manages the full lifecycle of a user query:
    1. Preprocessing (input validation and cleaning)
    2. Classification
    3. Aspect decomposition
    4. Specialized handling per aspect
    5. Response composition
    
    It maintains references to all specialized handlers and manages their execution.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                openai_model: str = "gpt-4o-mini",
                policy_api_base_url: Optional[str] = None):
        """
        Initialize the multi-aspect query processor
        
        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            openai_model (str): OpenAI model to use
            policy_api_base_url (Optional[str]): Base URL for the abortion policy API
        """
        logger.info("Initializing MultiAspectQueryProcessor")
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Use provided policy API URL or get from environment
        self.policy_api_base_url = policy_api_base_url or os.getenv("POLICY_API_BASE_URL", 
                                                                   "https://api.abortionpolicyapi.com/v1/")
        
        # Initialize preprocessing component
        self.preprocessor = Preprocessor()
        
        # Initialize components
        self.unified_classifier = UnifiedClassifier(api_key=self.api_key, model_name=openai_model)
        self.aspect_decomposer = AspectDecomposer(api_key=self.api_key, model_name=openai_model)
        self.response_composer = ResponseComposer()
        
        # Initialize specialized handlers
        self.handlers = {
            "knowledge": KnowledgeHandler(api_key=self.api_key, model_name=openai_model),
            "emotional": EmotionalSupportHandler(api_key=self.api_key, model_name=openai_model),
            "policy": PolicyHandler(api_key=self.api_key, policy_api_base_url=self.policy_api_base_url)
        }
        
        # Performance tracking
        self.processing_times = {
            "preprocessing": [],
            "classification": [],
            "decomposition": [],
            "handling": {},
            "composition": []
        }
        for handler_type in self.handlers.keys():
            self.processing_times["handling"][handler_type] = []
    
    async def process_query(self, 
                          message: str, 
                          conversation_history: List[Dict[str, Any]] = None,
                          user_location: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a user query through the multi-aspect pipeline
        
        Args:
            message (str): User query text
            conversation_history (List[Dict[str, Any]]): Previous conversation messages
            user_location (Optional[Dict[str, str]]): User's location data (state, city, etc.)
            
        Returns:
            Dict[str, Any]: Processed response
        """
        start_time = time.time()
        logger.info(f"Processing query: {message[:100]}...")
        
        try:
            # Initialize conversation history if None
            if conversation_history is None:
                conversation_history = []
            
            # 0. Preprocess the query
            preprocessing_start = time.time()
            preprocess_result = self.preprocessor.process(message)
            self.processing_times["preprocessing"].append(time.time() - preprocessing_start)
            
            logger.info(f"Preprocessing result: {preprocess_result['metadata']['preprocessing']}")
            
            # If message is not processable (e.g., non-English), return early
            if not preprocess_result['is_processable']:
                logger.info(f"Message not processable: {preprocess_result['stop_reason']}")
                return {
                    "text": preprocess_result['processed_message'],
                    "citations": [],
                    "citation_objects": [],
                    "preprocessing_metadata": preprocess_result['metadata'],
                    "processing_time": time.time() - start_time,
                    "timestamp": time.time(),
                    "message_id": str(uuid.uuid4()),
                    "session_id": getattr(self, "current_session_id", str(uuid.uuid4())),
                    "graphics": []
                }
            
            # Use the processed message for further processing
            processed_message = preprocess_result['processed_message']
            
            # 1. Classify the query
            classification_start = time.time()
            classification = await self.unified_classifier.classify(
                processed_message, 
                conversation_history
            )
            self.processing_times["classification"].append(time.time() - classification_start)
            
            logger.info(f"Classification result: {classification}")
            
            # 2. Decompose into aspects if needed
            decomposition_start = time.time()
            aspects = await self.aspect_decomposer.decompose(
                processed_message, 
                classification, 
                conversation_history
            )
            self.processing_times["decomposition"].append(time.time() - decomposition_start)
            
            logger.info(f"Decomposed into {len(aspects)} aspects")
            
            # 3. Process each aspect with specialized handlers
            aspect_responses = []
            aspect_tasks = []
            
            for aspect in aspects:
                aspect_type = aspect.get("type", "knowledge")
                
                if aspect_type in self.handlers:
                    # Create processing task for this aspect
                    handler = self.handlers[aspect_type]
                    task = asyncio.create_task(
                        self._process_aspect(
                            handler=handler,
                            aspect=aspect,
                            message=processed_message,
                            conversation_history=conversation_history,
                            user_location=user_location,
                            aspect_type=aspect_type,
                            original_message=message
                        )
                    )
                    aspect_tasks.append(task)
                else:
                    logger.warning(f"No handler available for aspect type: {aspect_type}")
            
            # Wait for all aspect processing to complete
            if aspect_tasks:
                aspect_responses = await asyncio.gather(*aspect_tasks)
                # Filter out None responses
                aspect_responses = [r for r in aspect_responses if r is not None]
            
            # 4. Compose the final response
            composition_start = time.time()
            response = self.response_composer.compose_response(
                message=processed_message,
                aspect_responses=aspect_responses,
                classification=classification
            )
            self.processing_times["composition"].append(time.time() - composition_start)
            
            # Add processing time to the response
            response["processing_time"] = time.time() - start_time
            
            # Add preprocessing metadata
            response["preprocessing_metadata"] = preprocess_result['metadata']
            
            # Transfer policy information to the main response if needed
            for aspect_response in aspect_responses:
                if aspect_response.get("aspect_type") == "policy":
                    # Copy primary content if it exists and our response doesn't have text
                    if "primary_content" in aspect_response and "text" not in response:
                        response["text"] = aspect_response["primary_content"]
                        logger.info("Using policy primary_content as text in response")
                    
                    # Also copy important policy fields to the main response
                    for key in ["state_code", "state_name", "policy_details", "policy_url", 
                               "policy_last_updated", "supportive_resources", "primary_content"]:
                        if key in aspect_response and key not in response:
                            response[key] = aspect_response[key]
                            logger.info(f"Copied policy field {key} to main response")
            
            # Store aspect responses for debugging/metrics
            response["aspect_responses"] = aspect_responses
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            # Return fallback response
            return {
                "text": "I apologize, but I encountered an error processing your question. Could you try rephrasing or asking a different question?",
                "citations": [],
                "citation_objects": [],
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "message_id": str(uuid.uuid4()),
                "session_id": getattr(self, "current_session_id", str(uuid.uuid4())),
                "graphics": [],
                "preprocessing_metadata": {"preprocessing": {"error": "internal_server_error"}}
            }
    
    async def _process_aspect(self, 
                             handler: Any, 
                             aspect: Dict[str, Any],
                             message: str,
                             conversation_history: List[Dict[str, Any]],
                             user_location: Optional[Dict[str, str]],
                             aspect_type: str,
                             original_message: str = None) -> Optional[Dict[str, Any]]:
        """
        Process a single aspect using the appropriate handler
        
        Args:
            handler: The specialized handler
            aspect: The aspect data
            message: Processed user message
            conversation_history: Previous conversation messages
            user_location: User's location data
            aspect_type: Type of the aspect
            original_message: Original user message before preprocessing
            
        Returns:
            Optional[Dict[str, Any]]: Handler response or None if failed
        """
        try:
            start_time = time.time()
            
            # Extract the aspect query
            aspect_query = aspect.get("query", message)
            confidence = aspect.get("confidence", 0.8)
            
            # Process query with specialized handler
            response = await handler.process_query(
                query=aspect_query,
                full_message=message,
                conversation_history=conversation_history,
                user_location=user_location
            )
            
            # If response is valid, add metadata
            if response and isinstance(response, dict):
                response["aspect_type"] = aspect_type
                response["confidence"] = confidence
                
                # Track processing time
                elapsed_time = time.time() - start_time
                self.processing_times["handling"][aspect_type].append(elapsed_time)
                
                # Make sure the aspect type is included in the response
                if aspect_type == "policy" and "state_code" in response:
                    # Format state info nicely if available
                    state_code = response.get("state_code")
                    state_name = response.get("state_name", "")
                    logger.info(f"Processed policy aspect for state: {state_code} ({state_name})")
                
                # Make sure the response has a non-empty primary_content
                if not response.get("primary_content"):
                    if aspect_type == "policy" and "state_code" in response:
                        # Generate a basic policy summary
                        state_name = response.get("state_name", state_code)
                        policy_details = response.get("policy_details", {})
                        summary = f"Here's abortion policy information for {state_name}:\n\n"
                        
                        # Add details if available
                        if policy_details:
                            for key, value in policy_details.items():
                                if key not in ["id", "state", "created_at", "updated_at"]:
                                    summary += f"• {key.replace('_', ' ').title()}: {value}\n"
                        else:
                            summary += "I'm sorry, but I don't have detailed policy information for this state. Please check official sources for the most accurate information."
                            
                        response["primary_content"] = summary
                    else:
                        response["primary_content"] = "I processed your question but couldn't generate a complete response. Please try asking in a different way."
                
                return response
            else:
                logger.warning(f"Handler for {aspect_type} returned invalid response")
                return None
                
        except Exception as e:
            logger.error(f"Error processing aspect {aspect_type}: {str(e)}", exc_info=True)
            return {
                "aspect_type": aspect_type,
                "primary_content": "I encountered an error processing your query. Please try again.",
                "error": str(e),
                "confidence": 0.0
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the query processor
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = {
            "average_times": {},
            "total_queries_processed": len(self.processing_times["classification"]),
            "handler_usage": {}
        }
        
        # Calculate average preprocessing time
        if self.processing_times["preprocessing"]:
            metrics["average_times"]["preprocessing"] = sum(self.processing_times["preprocessing"]) / len(self.processing_times["preprocessing"])
        
        # Calculate average times
        if self.processing_times["classification"]:
            metrics["average_times"]["classification"] = sum(self.processing_times["classification"]) / len(self.processing_times["classification"])
        
        if self.processing_times["decomposition"]:
            metrics["average_times"]["decomposition"] = sum(self.processing_times["decomposition"]) / len(self.processing_times["decomposition"])
        
        if self.processing_times["composition"]:
            metrics["average_times"]["composition"] = sum(self.processing_times["composition"]) / len(self.processing_times["composition"])
        
        # Calculate average times and usage for each handler
        for handler_type, times in self.processing_times["handling"].items():
            if times:
                metrics["average_times"][f"handling_{handler_type}"] = sum(times) / len(times)
                metrics["handler_usage"][handler_type] = len(times)
            else:
                metrics["handler_usage"][handler_type] = 0
        
        return metrics 