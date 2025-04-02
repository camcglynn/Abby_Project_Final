import logging
import uuid
import time
from typing import Dict, List, Any, Optional
import datetime
import re

logger = logging.getLogger(__name__)

class ResponseComposer:
    """
    Composes final responses by intelligently blending outputs from different specialized handlers
    """
    
    def __init__(self):
        """Initialize the response composer"""
        logger.info("Initializing ResponseComposer")
        
        # Define transition phrases for different aspect combinations
        self.transitions = {
            # Knowledge to Emotional
            "knowledge_emotional": [
                "In addition to this information, I want to acknowledge that ",
                "Beyond the facts, it's also important to consider that ",
                "While those are the medical facts, I understand that "
            ],
            
            # Knowledge to Policy
            "knowledge_policy": [
                "Regarding the legal aspects of this topic, ",
                "As for the policy implications, ",
                "In terms of the regulations in your area, "
            ],
            
            # Emotional to Knowledge
            "emotional_knowledge": [
                "To give you some additional information on this topic, ",
                "Here are some facts that might be helpful: ",
                "From a medical perspective, "
            ],
            
            # Emotional to Policy
            "emotional_policy": [
                "Regarding the legal situation, ",
                "As for the policies in your area, ",
                "In terms of the regulations that might affect you, "
            ],
            
            # Policy to Knowledge
            "policy_knowledge": [
                "Beyond the legal aspects, here's some general information: ",
                "In addition to the policy information, you might want to know that ",
                "From a medical perspective, "
            ],
            
            # Policy to Emotional
            "policy_emotional": [
                "I understand this information can bring up feelings. ",
                "Many people experience various emotions when considering these policies. ",
                "While those are the regulations, I recognize that this can be a complex emotional topic. "
            ]
        }
        
        # Define default transitions
        self.default_transition = "Additionally, "
    
    def compose_response(self, message: str, aspect_responses: List[Dict[str, Any]], 
                        classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compose a coherent response from multiple aspect handlers
        
        Args:
            message (str): Original user query
            aspect_responses (List[Dict[str, Any]]): Responses from specialized handlers
            classification (Dict[str, Any]): Original message classification
            
        Returns:
            Dict[str, Any]: Final composed response
        """
        try:
            if not aspect_responses:
                return self._create_fallback_response(message)
            
            # If there's just one response, return it directly with minimal processing
            if len(aspect_responses) == 1:
                main_response = aspect_responses[0]
                result_text = main_response.get('text', '')
                result_text = self._clean_text(result_text)
                
                return {
                    "text": result_text,
                    "citations": main_response.get('citations', []),
                    "citation_objects": main_response.get('citations', []),
                    "message_id": str(uuid.uuid4()),
                    "timestamp": time.time(),
                    **{k: v for k, v in main_response.items() if k not in ['text', 'citations', 'citation_objects']}
                }
            
            # For multiple aspects, we need to create a coherent combined response
            # First, organize responses by aspect type for better organization
            responses_by_type = {}
            for resp in aspect_responses:
                aspect_type = resp.get('aspect_type', 'knowledge')
                if aspect_type not in responses_by_type:
                    responses_by_type[aspect_type] = []
                responses_by_type[aspect_type].append(resp)
            
            # Sort responses within each type by confidence score
            for aspect_type in responses_by_type:
                responses_by_type[aspect_type] = sorted(
                    responses_by_type[aspect_type], 
                    key=lambda x: x.get('confidence', 0), 
                    reverse=True
                )
            
            # Determine presentation order: policy → knowledge → emotional
            # This order feels most natural in most conversations
            aspect_order = []
            for aspect_type in ['policy', 'knowledge', 'emotional']:
                if aspect_type in responses_by_type:
                    aspect_order.append(aspect_type)
            
            # Start constructing the response
            final_segments = []
            all_citations = []
            all_citation_objects = []  # Separate collection for citation objects
            
            # Track special attributes to include in the final response
            special_attributes = {}
            
            # Process each aspect type in order
            prev_aspect_type = None
            for i, aspect_type in enumerate(aspect_order):
                responses = responses_by_type[aspect_type]
                
                # Take the highest confidence response for this aspect type
                main_resp = responses[0]
                resp_text = main_resp.get('text', '').strip()
                
                # Skip empty responses
                if not resp_text:
                    continue
                
                # Add transition if this isn't the first segment
                if prev_aspect_type and i > 0:
                    transition_key = f"{prev_aspect_type}_{aspect_type}"
                    transition = self._get_transition(transition_key)
                    resp_text = f"{transition}{resp_text}"
                
                # Add to result
                final_segments.append(resp_text)
                prev_aspect_type = aspect_type
                
                # Add citations
                for citation in main_resp.get('citations', []):
                    if citation not in all_citations:
                        all_citations.append(citation)
                
                # Add citation objects if they exist
                if 'citation_objects' in main_resp and main_resp['citation_objects']:
                    for citation_obj in main_resp.get('citation_objects', []):
                        # Ensure we're dealing with a dictionary object, not a string
                        if isinstance(citation_obj, dict):
                            # Check if this citation object is already in our list (by URL)
                            is_duplicate = False
                            if citation_obj.get('url'):
                                for existing_obj in all_citation_objects:
                                    if isinstance(existing_obj, dict) and existing_obj.get('url') == citation_obj.get('url'):
                                        is_duplicate = True
                                        break
                            # If no URL, check by source name
                            else:
                                for existing_obj in all_citation_objects:
                                    if isinstance(existing_obj, dict) and existing_obj.get('source') == citation_obj.get('source'):
                                        is_duplicate = True
                                        break
                                        
                            if not is_duplicate:
                                all_citation_objects.append(citation_obj)
                        else:
                            # It's a string, so create a proper citation object
                            citation_source = str(citation_obj)
                            
                            # Check if we already have this source
                            is_duplicate = False
                            for existing_obj in all_citation_objects:
                                if isinstance(existing_obj, dict) and existing_obj.get('source') == citation_source:
                                    is_duplicate = True
                                    break
                                    
                            if not is_duplicate:
                                citation_dict = {
                                    "source": citation_source,
                                    "title": citation_source,
                                    "url": None,
                                    "accessed_date": datetime.datetime.now().strftime('%Y-%m-%d')
                                }
                                
                                # Handle Planned Parenthood with a default URL
                                if "Planned Parenthood" in citation_source:
                                    citation_dict["url"] = "https://www.plannedparenthood.org"
                                
                                all_citation_objects.append(citation_dict)
                
                # Collect special attributes (like state_code)
                if aspect_type == "policy":
                    if "state_code" in main_resp:
                        special_attributes["state_code"] = main_resp["state_code"]
                    if "state_codes" in main_resp:
                        special_attributes["state_codes"] = main_resp["state_codes"]
            
            # Combine all segments
            result_text = "\n\n".join(final_segments)
            
            # Clean up the text
            result_text = self._clean_text(result_text)
            
            # If no citation objects were found, but we have citations, create basic objects
            if not all_citation_objects and all_citations:
                for citation in all_citations:
                    if isinstance(citation, dict):
                        all_citation_objects.append(citation)
                    else:
                        # Create a basic citation object
                        citation_obj = {
                            "source": citation,
                            "accessed_date": datetime.datetime.now().strftime('%Y-%m-%d')
                        }
                        all_citation_objects.append(citation_obj)
            
            # Create the final response object
            response = {
                "text": result_text,
                "citations": all_citations,
                "citation_objects": all_citation_objects,
                "message_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "session_id": str(uuid.uuid4()),
                "graphics": [],
                **special_attributes
            }
            
            # Ensure the response has all required fields
            if "timestamp" not in response:
                response["timestamp"] = time.time()
            if "message_id" not in response:
                response["message_id"] = str(uuid.uuid4())
            if "session_id" not in response:
                response["session_id"] = str(uuid.uuid4())
            if "graphics" not in response:
                response["graphics"] = []
            
            return response
            
        except Exception as e:
            logger.error(f"Error composing response: {str(e)}", exc_info=True)
            return self._create_fallback_response(message)
    
    def _get_transition(self, transition_key: str) -> str:
        """
        Get an appropriate transition phrase
        
        Args:
            transition_key (str): Key for transition type
            
        Returns:
            str: Transition phrase
        """
        import random
        
        if transition_key in self.transitions:
            return random.choice(self.transitions[transition_key])
        
        return self.default_transition
    
    def _clean_text(self, text: str) -> str:
        """
        Clean response text by removing artifacts and redundancies
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        clean_text = ' '.join(text.split())
        
        # Remove redundant periods at the end
        clean_text = re.sub(r'\.+$', '.', clean_text)
        
        # Remove period before citation if one exists
        clean_text = re.sub(r'\.\s*\[\^', ' [^', clean_text)
        
        # Ensure proper spacing around citation markers
        clean_text = re.sub(r'\s+\[\^', ' [^', clean_text)
        clean_text = re.sub(r'\]\s+', '] ', clean_text)
        
        # Remove stray bracket citations like [.
        clean_text = re.sub(r'\[\.\s*', '', clean_text)
        clean_text = re.sub(r'\s?\[\.?\]', '', clean_text)
        
        # Clean up unnecessary citation descriptions
        clean_text = re.sub(r'\(Source:.*?\)', '', clean_text)
        
        # Ensure a period at the end of the text if it doesn't have one
        if not clean_text.rstrip().endswith(('.', '!', '?')):
            clean_text = clean_text.rstrip() + '.'

        return clean_text
    
    def _is_redundant_info(self, primary_text: str, secondary_text: str, 
                          threshold: float = 0.5) -> bool:
        """
        Check if secondary text contains redundant information already in primary text
        
        Args:
            primary_text (str): Main response text
            secondary_text (str): Additional response text
            threshold (float): Redundancy threshold
            
        Returns:
            bool: True if secondary text is redundant
        """
        # Simple word overlap for redundancy check
        primary_words = set(primary_text.lower().split())
        secondary_words = set(secondary_text.lower().split())
        
        if not secondary_words:
            return True
        
        # Calculate overlap
        overlap = len(primary_words.intersection(secondary_words)) / len(secondary_words)
        
        # Check if significant portion of the secondary text is already in primary
        return overlap > threshold
    
    def _create_fallback_response(self, message: str) -> Dict[str, Any]:
        """
        Create a fallback response when something goes wrong
        
        Args:
            message (str): The original user query
            
        Returns:
            Dict[str, Any]: Fallback response
        """
        message_id = str(uuid.uuid4())
        
        return {
            "text": "I apologize, but I couldn't generate a complete response to your question. Could you try rephrasing or asking another question about reproductive health?",
            "citations": [],
            "citation_objects": [],
            "message_id": message_id,
            "timestamp": time.time(),
            "session_id": str(uuid.uuid4()),
            "graphics": [],
            "preprocessing_metadata": {"preprocessing": {"error": "fallback_response"}}
        } 