import os
import logging
import json
import time
from openai import OpenAI
from chatbot.config import config
from typing import List, Dict, Any, Optional
import tiktoken
from utils.metrics import record_api_call, record_time
from utils.text_processing import PIIDetector
import random
import openai

logger = logging.getLogger(__name__)


class GPTModel:
    """
    Integration with OpenAI's GPT-4 for enhanced conversational responses
    with efficient token usage
    """

    def __init__(self):
        """Initialize the GPT integration"""
        logger.info("Initializing GPT Model")
        
        # Initialize default values to ensure they exist even if initialization fails
        self.openai_available = False
        self.api_key_valid = False
        self.use_cache = True
        self.response_cache = {}
        self.max_cache_size = 200
        self.bert_rag_model = None
        self.current_token_usage = 0
        self.monthly_token_budget = int(os.environ.get("MONTHLY_TOKEN_BUDGET", "1000000"))
        self.token_usage_file = "logs/token_usage.json"
        
        try:
            # Get API key
            api_key = config.get_api_key("openai")
            logger.info(f"Using API key starting with: {api_key[:5]}... (length: {len(api_key)})")
            
            # Initialize the OpenAI client
            self.client = OpenAI(api_key=api_key)
            self.openai_available = True
            self.api_key_valid = True
            
            # Get model settings
            model_settings = config.get_model_settings()
            self.model = model_settings["model"] 
            self.temperature = model_settings["temperature"]
            self.max_tokens = model_settings["max_tokens"]
            logger.info(f"Using model: {self.model}, temperature: {self.temperature}, max_tokens: {self.max_tokens}")
            
            # Initialize tokenizer for token counting
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except Exception as tok_err:
                logger.error(f"Error initializing tokenizer: {str(tok_err)}")
                # Fallback to a common tokenizer
                logger.info("Falling back to cl100k_base tokenizer")
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            # Define system prompt
            self.system_prompt = """You are a highly empathetic, caring, and knowledgeable reproductive health assistant. 
            Your role is to provide accurate, non-judgmental information while being deeply sensitive to the personal 
            nature of these topics. Always prioritize user privacy, emotional well-being, and safety. Connect with users on a 
            personal level, acknowledge their feelings, and provide reassurance throughout your responses. If you're unsure about something, 
            acknowledge it and suggest consulting healthcare professionals. Base your responses on verified medical 
            information and current policy data.
            
            Important: Consider the full conversation context when responding. Reference previous messages when
            appropriate, and maintain a warm, supportive conversation flow. If the user asks a follow-up question or
            references something from earlier in the conversation, make sure to acknowledge and address this with care.
            
            If the user provides a zip code, use it to provide location-specific information. If they ask
            about abortion access and provide a zip code, give them specific information about clinics and
            policies in their area with a compassionate tone that validates their situation and concerns."""
            
            # Cache system prompt tokens
            self.system_prompt_tokens = len(self.tokenizer.encode(self.system_prompt))
            
            # Configure models for tiered usage (cheaper models for simpler tasks)
            self.model_tiers = {
                "cheap": "gpt-3.5-turbo",   # For simple queries and classification
                "standard": "gpt-3.5-turbo", # For most responses
                "premium": self.model        # For complex or policy questions
            }
            
            # Try to verify available models if API is accessible
            try:
                # Log available models
                all_models = self.client.models.list()
                available_model_ids = [m.id for m in all_models.data]
                logger.info(f"Available models: {available_model_ids[:5]}...")
                
                # Update model tiers to ensure they exist
                for tier, model_id in self.model_tiers.items():
                    if model_id not in available_model_ids and model_id != self.model:
                        logger.warning(f"Model {model_id} for tier {tier} not found in available models. Falling back to gpt-3.5-turbo")
                        self.model_tiers[tier] = "gpt-3.5-turbo"
            except Exception as model_err:
                logger.error(f"Error checking available models: {str(model_err)}")
                # Continue with default models
            
            # Load token usage data
            self._load_token_usage()
            
            logger.info("GPT Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing GPT Model: {str(e)}")
            # Continue initialization with defaults already set

    def _load_token_usage(self):
        """Load token usage from file"""
        try:
            if os.path.exists(self.token_usage_file):
                with open(self.token_usage_file, 'r') as f:
                    usage_data = json.load(f)
                    self.current_token_usage = usage_data.get('current_usage', 0)
                    logger.info(f"Loaded token usage: {self.current_token_usage}/{self.monthly_token_budget}")
            else:
                # Initialize with zero if file doesn't exist
                os.makedirs(os.path.dirname(self.token_usage_file), exist_ok=True)
                with open(self.token_usage_file, 'w') as f:
                    json.dump({'current_usage': 0}, f)
        except Exception as e:
            logger.error(f"Error loading token usage: {str(e)}")
            # Continue with zero usage in case of error
            self.current_token_usage = 0
            
    def _save_token_usage(self):
        """Save current token usage to file"""
        try:
            with open(self.token_usage_file, 'w') as f:
                json.dump({'current_usage': self.current_token_usage}, f)
        except Exception as e:
            logger.error(f"Error saving token usage: {str(e)}")
            
    def _update_token_usage(self, prompt_tokens, completion_tokens):
        """Update token usage counters"""
        total_tokens = prompt_tokens + completion_tokens
        self.current_token_usage += total_tokens
        self._save_token_usage()
        
        # Log warning if approaching budget
        budget_used_percent = (self.current_token_usage / self.monthly_token_budget) * 100
        if budget_used_percent > 80:
            logger.warning(f"Token budget usage high: {budget_used_percent:.1f}% ({self.current_token_usage}/{self.monthly_token_budget})")
            
        return total_tokens

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def _prepare_messages(self, 
                         question: str, 
                         history: Optional[List[Dict[str, Any]]] = None,
                         max_history_tokens: int = 1000) -> List[Dict[str, str]]:
        """
        Prepare messages for API call with token management
        
        Args:
            question (str): Current question
            history (Optional[List[Dict[str, Any]]]): Conversation history
            max_history_tokens (int): Maximum tokens to use for history
            
        Returns:
            List[Dict[str, str]]: Prepared messages
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Calculate available tokens for history
        question_tokens = self._count_tokens(question)
        available_tokens = max_history_tokens
        
        # Add relevant history if provided
        if history:
            history_messages = []
            for msg in reversed(history):  # Process from most recent
                content = msg['message']
                msg_tokens = self._count_tokens(content)
                
                if available_tokens - msg_tokens > 0:
                    history_messages.insert(0, {
                        "role": "user" if msg['sender'] == 'user' else "assistant",
                        "content": content
                    })
                    available_tokens -= msg_tokens
                else:
                    break
            
            messages.extend(history_messages)
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        return messages

    def _create_cache_key(self, question, history=None):
        """Create a cache key from a question and optional history"""
        # Start with the question
        key = question.lower().strip()
        
        # If there's history, add a hash of the last message to the key
        if history and len(history) > 0:
            last_message = history[-1].get('message', '')
            # Use only last user message to avoid cache misses due to timestamps, etc.
            key += "_" + str(hash(last_message.lower().strip()))
            
        return key

    def get_response(self, question: str, history: List[Dict] = None) -> str:
        """
        Get a response from the GPT model, intelligently handling errors and PII.
        
        Args:
            question: The user's question
            history: Conversation history
            
        Returns:
            str: The model's response
        """
        try:
            # Initialize PII detector if not already done
            pii_detector = PIIDetector()
            
            logger.info("Getting GPT response for question")
            
            # First check for PII in the question
            has_pii = pii_detector.has_pii(question)
            if has_pii:
                logger.warning("PII detected in question, sanitizing before sending to OpenAI")
                sanitized_question, _ = pii_detector.redact_pii(question)
                # Note: This will preserve zip codes and state names due to our updated PII detector
                question = sanitized_question
            
            # Also check for PII in the history (last 10 messages)
            if history and len(history) > 0:
                sanitized_history = []
                for msg in history[-10:]:  # Only examine last 10 messages
                    if msg.get('sender', '') == 'user':
                        message_text = msg.get('message', '')
                        if pii_detector.has_pii(message_text):
                            logger.warning(f"PII detected in history message {msg.get('message_id', 'unknown')}, sanitizing")
                            sanitized_text, _ = pii_detector.redact_pii(message_text)
                            msg_copy = msg.copy()
                            msg_copy['message'] = sanitized_text
                            sanitized_history.append(msg_copy)
                        else:
                            sanitized_history.append(msg)
                    else:
                        sanitized_history.append(msg)
                history = sanitized_history
                
            # Check cache first if enabled
            if self.use_cache:
                cache_key = self._create_cache_key(question, history)
                cached_response = self.response_cache.get(cache_key)
                if cached_response:
                    logger.info("Returning cached response")
                    return cached_response
                    
            # Select appropriate model based on question complexity
            model_to_use = self._select_appropriate_model(question, history)
            
            # Prepare messages with system prompt
            messages = self._prepare_messages(question, history)
            
            # Get response from OpenAI
            logger.info(f"Sending request to OpenAI with model: {model_to_use}")
            
            # Track start time for performance monitoring
            start_time = time.time()
            
            try:
                # Get response
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Calculate time taken
                time_taken = time.time() - start_time
                logger.info(f"GPT response received in {time_taken:.2f} seconds")
                
                # Extract response text  
                completion_text = response.choices[0].message.content
                
                # Track token usage metrics  
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = self._update_token_usage(prompt_tokens, completion_tokens)
                
                # Record metrics
                record_api_call(f"openai_{model_to_use}", total_tokens)
                record_time("openai_response_time", time_taken)
                
                # Log usage
                logger.info(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                
                # Cache the response if caching is enabled
                if self.use_cache:
                    self.response_cache[cache_key] = completion_text
                    # Prune cache if it gets too large
                    if len(self.response_cache) > self.max_cache_size:
                        # Remove a random item
                        random_key = random.choice(list(self.response_cache.keys()))
                        del self.response_cache[random_key]
                        
                return completion_text
                
            except Exception as api_error:
                logger.error(f"Error in OpenAI API call: {str(api_error)}")
                
                # Try to fallback to bert_rag model
                try:
                    from chatbot.bert_rag import BertRAGModel
                    
                    if not self.bert_rag_model:
                        self.bert_rag_model = BertRAGModel()
                        
                    rag_response = self.bert_rag_model.get_response(question)
                    logger.info("Using fallback response from BERT RAG model")
                    return rag_response
                except Exception as fallback_error:
                    logger.error(f"Error in fallback response generation: {str(fallback_error)}")
                    return self._get_fallback_response(question)
        
        except Exception as e:
            logger.error(f"Error in GPT response generation: {str(e)}")
            logger.error(str(e), exc_info=True)
            return "I'm sorry, I encountered an error with that question. Could you try rephrasing it?"

    def _select_appropriate_model(self, question, history=None):
        """
        Select the most cost-effective model based on question complexity
        
        Args:
            question (str): The user question
            history (Optional[List]): Conversation history
            
        Returns:
            str: Model name to use (e.g., 'gpt-3.5-turbo')
        """
        question_lower = question.lower()
        
        # Use cheap model for very simple queries
        if len(question.split()) <= 5:
            # Simple greetings, thanks, etc.
            simple_phrases = ["hi", "hello", "hey", "thanks", "thank you", "goodbye", "bye"]
            if any(phrase in question_lower for phrase in simple_phrases):
                return self.model_tiers.get("cheap", "gpt-3.5-turbo")
        
        # Default to standard for most queries
        # Only use premium for very specific complex queries
        # This is a major cost-saving change - previously many queries would use premium
        
        # Only use premium for complex policy or medical questions
        if "policy" in question_lower and "abortion" in question_lower:
            # Check if it has complex elements indicating need for premium
            complex_indicators = ["compare", "difference", "ethical", "debate", "controversy", "complex"]
            if any(indicator in question_lower for indicator in complex_indicators):
                return self.model_tiers.get("premium", self.model)
        
        # Use standard model for everything else to save costs
        return self.model_tiers.get("standard", "gpt-3.5-turbo")
        
    def _get_fallback_response(self, question):
        """
        Generate a fallback response when over budget
        
        Args:
            question (str): The user question
            
        Returns:
            str: A fallback response
        """
        # Generic fallback response
        generic_fallback = ("I apologize, but we're currently experiencing high demand. "
                          "For immediate assistance with reproductive health questions, "
                          "please contact Planned Parenthood at 1-800-230-PLAN or visit www.plannedparenthood.org.")
        
        # Try to provide a more specific response based on question patterns
        question_lower = question.lower()
        
        if "abortion" in question_lower and any(word in question_lower for word in ["legal", "law", "allowed", "state"]):
            return (f"For up-to-date information on abortion laws in your state, please visit "
                   f"www.ineedana.com or contact Planned Parenthood at 1-800-230-PLAN.")
                   
        if "clinic" in question_lower or "where can i" in question_lower:
            return ("To find reproductive healthcare providers near you, please visit "
                   "www.ineedana.com or www.plannedparenthood.org/health-center")
                   
        if "cost" in question_lower or "insurance" in question_lower or "pay" in question_lower:
            return ("For information about costs and insurance coverage for reproductive healthcare, "
                   "please contact your local Planned Parenthood health center at 1-800-230-PLAN.")
                   
        return generic_fallback

    def get_chat_response(self, message, history=None):
        """
        Get a chat response for conversational messages.
        This is a wrapper around generate_response with appropriate formatting.
        
        Args:
            message (str): The user's message
            history (List[Dict], optional): Conversation history
            
        Returns:
            str: The assistant's response
        """
        try:
            logger.info(f"Getting chat response for: {message[:30]}...")
            
            # For simple chat interactions, use the generate_response method
            conversational_system_message = """
            You're Abby, a warm and empathetic reproductive health assistant.
            Keep your responses friendly, supportive, and conversational.
            Be warm, reassuring, and non-judgmental.
            Use natural language like you're having a friendly chat.
            """
            
            response = self.generate_response(
                prompt=message,
                system_message=conversational_system_message,
                temperature=0.7
            )
            
            logger.info(f"Generated chat response: {response[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error in get_chat_response: {str(e)}", exc_info=True)
            return "I'm sorry, I'm having trouble responding right now. Could you rephrase your question?"

    def detect_policy_question(self, question, conversation_history=None):
        """
        Use GPT to detect if a question is about abortion policy/access in a specific state

        Args:
            question (str): User's question
            conversation_history (list, optional): Previous conversation messages

        Returns:
            bool: True if the question is about abortion policy, False otherwise
        """
        try:
            # If we've already determined this is a policy question, don't waste tokens
            if "abortion" in question.lower() and any(
                    term in question.lower()
                    for term in ["legal", "allowed", "can i get", "access"]):
                return True

            # For more subtle questions, especially those with referential terms, use GPT
            history_context = ""
            if conversation_history:
                # Get the last few user messages for context
                user_messages = [
                    msg['message'] for msg in conversation_history
                    if msg['sender'] == 'user'
                ][-3:]
                if user_messages:
                    history_context = "Previous messages:\n" + "\n".join(
                        user_messages)

            prompt = f"""
            Analyze this question to determine if it's about abortion access, legality, or policy in a specific state.
            Return ONLY "yes" if it's asking about state-specific abortion policy/access, or "no" otherwise.

            {history_context}

            Question: {question}

            Is this about state-specific abortion policy/access (yes/no):
            """

            response = self.get_response(prompt).strip().lower()
            return "yes" in response

        except Exception as e:
            logger.error(f"Error detecting policy question: {str(e)}")
            # Default to False on error
            return False

    def enhance_response(self, 
                        question: str, 
                        rag_response: str,
                        max_output_tokens: int = 500) -> str:
        """
        Enhance a RAG response using GPT for better quality and empathy,
        with efficient token usage
        
        Args:
            question (str): User's question
            rag_response (str): Response from the RAG system
            max_output_tokens (int): Maximum tokens for output

        Returns:
            str: Enhanced response
        """
        try:
            # Calculate available tokens
            prompt_template = """
            Transform this response into a deeply empathetic, supportive, and comprehensive answer that connects with the person on a human level:
            
            Question: {question}
            Original response: {response}
            
            Requirements:
            Create a warm, conversational tone as if talking to a friend who needs support
            Begin with genuine acknowledgment of feelings and validate their experience
            Use caring and supportive language throughout
            Maintain factual accuracy while adding emotional depth
            Show understanding of the personal impact this topic might have
            
            Format Guidelines:
            Start with an empathetic opening that connects with their situation
            Use personal language like "I understand," "I hear you," and "You're not alone"
            Present information in a conversational, supportive flow
            Add reassurance and validation throughout
            End with a supportive statement that offers hope
            Add markdown bold syntax (**text**) for important terms
            Show the actual ** symbols in the output, don't render them as bold
            
            The response should feel like it's coming from a caring, knowledgeable friend who understands both the facts and the emotions involved.
            """
            
            formatted_prompt = prompt_template.format(
                question=question,
                response=rag_response
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=max_output_tokens)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error enhancing response: {str(e)}", exc_info=True)
            return rag_response  # Fall back to original response

    def format_policy_response(self, 
                             question: str, 
                             state: str, 
                             policy_data: Dict[str, Any],
                             max_output_tokens: int = 600) -> str:
        """
        Format policy API data into a user-friendly response using GPT
        with efficient token usage
        
        Args:
            question (str): User's question
            state (str): State name
            policy_data (Dict[str, Any]): Policy data
            max_output_tokens (int): Maximum tokens for output

        Returns:
            str: Formatted response
        """
        try:
            # Extract only necessary policy data to save tokens
            essential_data = self._extract_essential_policy_data(policy_data)
            
            prompt = f"""
            Create a deeply empathetic and supportive response about abortion policies in {state}.
            Question: {question}
            Policy data: {json.dumps(essential_data)}
            
            Requirements:
- Format the response as a numbered list
- Each policy aspect should be a separate, clearly numbered line item
- Provide clear, concise information
- Use markdown bold for key terms
- Maintain an empathetic and supportive tone
- Ensure each point is standalone and informative

Policy Information Context:
- Abortion policy details for {state}
- Question asked: {question}
- Available policy data: {json.dumps(essential_data)}

Expected Output Format:
1. First policy point with detailed explanation
2. Second policy point with detailed explanation
3. Third policy point with detailed explanation
... and so on

Specific Guidelines:
- Use direct, clear language
- Include practical implications of each policy point
- Show compassion and understanding in the explanation
- Avoid legal jargon
- Make the information easily digestible
            
            Tone Examples:
            Instead of "Abortion is legal until 24 weeks" use "I want you to know that in your state, abortion is legally accessible until 24 weeks of pregnancy, giving you time to consider your options carefully."
            Instead of "There is a 24-hour waiting period" use "I understand waiting periods can be frustrating and add complexity to an already difficult decision. In your state, there is a required 24-hour waiting period between consultation and procedure."
            """

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,
                max_tokens=max_output_tokens)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error formatting policy response: {str(e)}", exc_info=True)
            return f"I have information about {state}, but I'm having trouble formatting it right now. I understand this may be frustrating when you're looking for important information. Please try asking a more specific question, or I can try connecting you with direct resources that might help."

    def _extract_essential_policy_data(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only essential policy data to reduce tokens"""
        essential = {}
        if "endpoints" in policy_data:
            endpoints = policy_data["endpoints"]
            if "gestational_limits" in endpoints:
                essential["gestational_limits"] = endpoints["gestational_limits"]
            if "waiting_periods" in endpoints:
                essential["waiting_periods"] = endpoints["waiting_periods"]
        return essential

    def generate_response(self,
                          prompt,
                          messages=None,
                          system_message=None,
                          temperature=0.8):
        """Generate a conversational response using the OpenAI API."""
        try:
            if messages is None:
                # Default to a friendly system message
                if system_message is None:
                    system_message = """
                    You're Abby, a warm and caring reproductive health assistant.
                    Speak naturally, like a friend who's knowledgeable but never robotic.
                    - Be warm, reassuring, and non-judgmental.
                    - Keep responses natural and engaging, like a chat with a supportive friend.
                    - Use simple, conversational language, avoiding rigid structures.
                    - If unsure, acknowledge it rather than making up information.
                    - Ask clarifying questions when necessary to keep the conversation going.
                    """

                messages = [{
                    "role": "system",
                    "content": system_message
                }, {
                    "role": "user",
                    "content": prompt
                }]

            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                max_tokens=1000,
                temperature=temperature)

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating response from GPT: {str(e)}")
            return "Oops! I'm having a little trouble right now. Mind trying again in a bit?"
