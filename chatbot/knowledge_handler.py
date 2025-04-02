import logging
import os
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import re
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from datetime import datetime
import json
from openai import OpenAI

# Import for OpenAI
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)

class KnowledgeHandler:
    """
    Handler for knowledge-seeking aspects of user queries.
    
    This class processes factual questions about reproductive health,
    accessing reliable knowledge sources and generating informative responses.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        """
        Initialize the knowledge handler
        
        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            model_name (str): OpenAI model to use
        """
        logger.info(f"Initializing KnowledgeHandler with model {model_name}")
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Set up OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model_name
        
        # Initialize data structures
        self.data = pd.DataFrame()
        self.index = None
        self.embeddings = None
        
        # Try to load BERT model for embeddings
        try:
            from transformers import AutoModel, AutoTokenizer
            self.embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Successfully loaded BERT model for embeddings")
            
            # Load datasets and build index
            self.data = self._load_datasets()
            self.index, self.embeddings = self._build_index()
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            self.embedding_model = None
            self.tokenizer = None
        
        # Knowledge response prompt template
        self.knowledge_prompt = """

You are a knowledgeable and compassionate reproductive health specialist providing factual information in an empathetic tone.
User query: {query}
Full message context: {full_message}
Using the following knowledge sources to inform your response:
{knowledge_sources}

--- OUTPUT STRUCTURE REQUIREMENTS ---
Generate the response adhering STRICTLY to the following structure and guidelines:

1.  **Introduction:** Start with a SINGLE introductory sentence that acknowledges the query's topic sensitively and summarizes the main point briefly.
2.  **Section Delimiter:** Follow the introduction with a blank line.
3.  **Section Content & Tone:** For each distinct point or aspect covered in the knowledge sources relevant to the query:
    *   Start a new section with a delimiter `###TITLE_TEXT###` where TITLE_TEXT is a concise, descriptive title for the section (e.g., `###What is Abortion###`, `###Types of Procedures###`). The delimiter MUST be on its own line.
    *   Follow the delimiter line IMMEDIATELY with the paragraph(s) explaining that section's topic.
    *   **CONTENT SOURCE:** Use information **ONLY** from the provided knowledge sources. Do not add outside information or opinions.
    *   **TONE (CRITICAL):** Balance **empathy** with **factual accuracy**. Use clear, sensitive, supportive, and non-judgmental language. Avoid overly clinical or alarming terms where possible, but prioritize medical accuracy. Ensure the tone is helpful and understanding.
    *   **MEDICAL TERMS:** Use **bold formatting** for key medical terms when first introduced within the explanation text itself (NOT in the title delimiter).
    *   **SEPARATION:** Separate each `###TITLE###` block (delimiter + text) from the next with a SINGLE blank line.
4.  **EXCLUSIONS:** **DO NOT** add your own "Sources:", "References:", "Citations:", "Disclaimer:", or any similar list/section at the end of your response. Citation information is handled separately. Do not add conversational filler.

--- EXAMPLE OF REQUIRED OUTPUT FORMAT ---
This is the single introductory sentence about the topic, written sensitively.

###First Section Title###
This is the paragraph explaining the first section, balancing empathy and facts. Use **key terms** here. Based only on provided sources.

###Second Section Title###
This paragraph explains the second section, written clearly and non-judgmentally. Based only on provided sources.

###Third Section Title###
This paragraph explains the third section with accurate details in a supportive tone. Based only on provided sources.
--- END OF EXAMPLE ---

Generate the response now using ONLY the provided sources and adhering STRICTLY to the delimiter-based structure (`###TITLE###\nContent\n\n`), the tone requirements, and the instruction to NOT add a sources list or disclaimers.
"""
    async def process_query(self, query: str, full_message: Optional[str] = None,
                      conversation_history: Optional[List[Dict[str, Any]]] = None,
                      user_location: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process query, generate HTML response via post-processing, use modified citation logic
        to create "Source: Title" display format.
        """
        start_time = time.time()
        logger.info(f"Processing knowledge query (V3.1 -> HTML, Combined Title Citations): {query[:100]}...")
        response_text = "" # Final HTML response
        docs = []          # Docs from vector search
        scores = []
        final_confidence = 0.2
        citation_objects = [] # Initialize for citation logic
        citation_sources = [] # Initialize for citation logic

        try:
            query_text = query or full_message
            if not query_text:
                raise ValueError("No query text provided")

        # --- 1. Retrieve Context ---
            can_retrieve = self.index is not None # ... and other checks ...
            if can_retrieve:
                try:
                # Ensure _retrieve_context returns dicts with 'source', 'url', 'title'(optional), 'question'(optional), 'answer'
                    docs, scores = await self._retrieve_context(query_text, top_k=5)
                    if not isinstance(docs, list): docs = []
                except Exception as retrieve_err:
                    logger.error(f"Error during context retrieval: {retrieve_err}", exc_info=True)
                    docs = []
            else:
                logger.warning("Skipping vector search...")
                docs = []

        # --- Response Generation & Citation Creation ---
            if docs and len(docs) > 0:
                logger.info(f"Found {len(docs)} relevant documents...")

            # --- 2a. Generate Response (OpenAI + Post-processing) ---
                if self.client:
                    formatted_sources = self._format_vector_sources(docs) # Ensure this helper exists
                    prompt_content = self.knowledge_prompt.format( # Uses V3.1 prompt
                        query=query_text, full_message=full_message or query_text,
                        knowledge_sources=formatted_sources
                )
                    try:
                        logger.debug("Sending request to OpenAI API (V3.1 structure)...")
                    # Use asyncio.to_thread for blocking client call
                    # If using AsyncOpenAI client, just use 'await self.client.chat.completions.create(...)'
                        response = await asyncio.to_thread(
                            self.client.chat.completions.create,
                            model=self.model, messages=[
                                {"role": "system", "content": "You are a knowledgeable... assistant..."},
                                {"role": "user", "content": prompt_content}
                            ], temperature=0.2
                    )
                        raw_response_text = response.choices[0].message.content.strip()
                        logger.debug(f"Raw OpenAI response received:\n{raw_response_text}")

                    # --- Cleanup Step: Remove potential trailing "Sources:" block ---
                        cleaned_raw_response = re.split(r'\n\s*(Sources:|Citations:|References:)\s*\[?\d?.*', raw_response_text, flags=re.IGNORECASE | re.DOTALL)[0].strip()
                        if len(cleaned_raw_response) < len(raw_response_text): logger.info("Removed potential trailing 'Sources:' block added by LLM.")
                        else: logger.debug("No trailing 'Sources:' block found or removed.")

                    # --- HTML Post-processing ---
                        html_parts = []
                        nl = '\n'
                        sections = re.split(fr'{nl}\s*###(.*?)###\s*{nl}', cleaned_raw_response, flags=re.DOTALL)
                        if len(sections) > 1 and len(sections) % 2 != 0:
                            intro = sections[0].strip()
                            if intro: html_parts.append(f"<p>{intro.replace(nl, '<br>')}</p>")
                            for i in range(1, len(sections), 2):
                                title = sections[i].strip(); content = sections[i+1].strip()
                                if title and content:
                                    content_html = content.replace(nl, '<br>')
                                    html_parts.append(f'<p><strong>{title}:</strong><br>{content_html}</p>')
                            if html_parts:
                                response_text = "".join(html_parts)
                                final_confidence = 0.85
                            else:
                                response_text = f"<p>{cleaned_raw_response.replace(nl, '<br>')}</p>"; final_confidence = 0.6
                        else:
                            response_text = f"<p>{cleaned_raw_response.replace(nl, '<br>')}</p>"; final_confidence = 0.6

                    except Exception as api_err:
                        logger.error(f"Error calling OpenAI API or post-processing: {api_err}", exc_info=True)
                        response_text = f"<p>{self._format_simple_response(docs)}</p>"
                        final_confidence = 0.4
                else: # No OpenAI client
                    logger.warning("OpenAI client not configured...")
                    response_text = f"<p>{self._format_simple_response(docs)}</p>"
                    final_confidence = 0.5

            # --- 2b. Create Citations (MODIFIED BLOCK FOR DISPLAY TITLE) ---
                logger.info(f"Generating citations with combined title format (up to 3)...")
            # citation_objects = [] # Initialized at top
                for i, doc in enumerate(docs):
                    if i >= 3:  # Limit to the top 3 most relevant sources
                        break

                    if not isinstance(doc, dict):
                        logger.warning(f"Skipping citation for doc index {i}: Item not a dict ({type(doc)}).")
                        continue

                    source = doc.get("source", "Unknown Source")
                    url = doc.get("url", "") # Ensure 'url' key exists

                # Only add citations with URLs
                    if url and isinstance(url, str) and url.startswith(('http://', 'https://')):
                    # --- Start of User's Title Derivation Logic ---
                        derived_title = doc.get("title", "") # Check for explicit 'title' key first
                        if not derived_title and url: # If no 'title' key, parse URL
                            try:
                                parsed_url = urlparse(url)
                                path_parts = parsed_url.path.strip('/').split('/')
                                if path_parts and path_parts[-1]:
                                    raw_title = path_parts[-1].replace('-', ' ').replace('_', ' ')
                                    derived_title = ' '.join(word.capitalize() for word in raw_title.split() if word)
                            # If path parsing doesn't yield title, derived_title remains empty
                            except Exception as e:
                                logger.warning(f"Error extracting title from URL '{url}': {e}")
                                derived_title = "" # Ensure it's empty on error
                    # --- End of User's Title Derivation Logic ---

                    # *** Construct the final display title ***
                        cleaned_derived_title = derived_title.strip() if derived_title else ""

                        if cleaned_derived_title and cleaned_derived_title.lower() != source.lower():
                        # Combine source and derived title if derived title is meaningful and different
                            final_display_title = f"{source}: {cleaned_derived_title}"
                        else:
                        # Otherwise, just use the source name as the title
                            final_display_title = source

                        logger.debug(f"Citation {i}: Source='{source}', Derived='{cleaned_derived_title}', Final Title='{final_display_title}'")

                        citation_obj = {
                        "source": source, # Keep original source separate if needed elsewhere
                        "url": url,
                        "title": final_display_title, # Store the combined/final title here
                        "accessed_date": datetime.now().strftime('%Y-%m-%d')
                    }
                    # Prevent duplicate citations based on URL
                        if url not in [c['url'] for c in citation_objects]:
                            citation_objects.append(citation_obj)
                        else:
                            logger.debug(f"Skipping duplicate citation for URL: {url}")


            # Create simple citation sources list (remains the same)
                citation_sources = [c["source"] for c in citation_objects]

            # Log the citations with details (now logging the potentially combined title)
                logger.info(f"Generated {len(citation_objects)} citation objects (combined title):")
                for c in citation_objects:
                    logger.info(f"  - Source: {c.get('source', 'N/A')}, URL: {c.get('url', 'N/A')}, Title Field: {c.get('title', 'N/A')}")

            else: # Fallback if no docs were retrieved
                logger.warning("No relevant documents found for query")
                response_text = "<p>I couldn't find specific information... consult a healthcare provider.</p>" # HTML wrap
                final_confidence = 0.3
            # Citations remain empty

        # --- 4. Return Result ---
            processing_time = time.time() - start_time
            logger.info(f"Knowledge response processing finished in {processing_time:.2f} seconds")

            return {
            "text": response_text.strip(),        # The final HTML text
            "citations": citation_sources,       # List of source names
            "citation_objects": citation_objects,# List of detailed dicts with combined title
            "aspect_type": "knowledge",
            "confidence": final_confidence,
            "processing_time": processing_time
        }

    # --- Error Handling ---
        except ValueError as ve:
            logger.error(f"Value error: {ve}")
            processing_time = time.time() - start_time if 'start_time' in locals() else -1
            return { "text": f"<p>Error: {ve}.</p>", "citations": [], "citation_objects": [], "aspect_type": "knowledge", "confidence": 0.1, "processing_time": processing_time if processing_time != -1 else None }
        except Exception as e:
            logger.error(f"Critical unexpected error: {e}", exc_info=True)
            processing_time = time.time() - start_time if 'start_time' in locals() else -1
            return { "text": "<p>I encountered an unexpected error...</p>", "citations": [], "citation_objects": [], "aspect_type": "knowledge", "confidence": 0.1, "processing_time": processing_time if processing_time != -1 else None }

    def _find_relevant_sources(self, query: str) -> List[Dict[str, Any]]:
        """Find relevant knowledge sources for query

        Args:
            query (str): The user query

        Returns:
            List[Dict[str, Any]]: List of relevant knowledge sources
        """
        return []

    def _format_knowledge_sources(self, sources: List[Dict[str,Any]]) -> str:
        """Format knowledge sources for inclusion in the prompt.

        Args:
            sources (List[Dict[str,Any]]): Knowledge sources to format

        Returns:
            str: Formatted knowledge sources text
        """
        if not sources:
            return "No specific knowledge sources available for this query."
        formatted_text = ""
        for source in sources:
            formatted_text += f"SOURCE: {source.get('source', 'Unknown Source')}\n"
            formatted_text += f"{source.get('answer', '')}\n\n"
        return formatted_text

    def _format_vector_sources(self, docs: List[Dict[str, Any]]) -> str:
        """
        Format vector search results for inclusion in the prompt
        (Based on user's original code structure)
        """
        if not docs:
            return "No specific knowledge sources available for this query."

        formatted_text = "" 

        for i, doc in enumerate(docs):
             formatted_text += f"SOURCE [{i+1}]: {doc.get('source', 'Unknown Source')}\n"
             formatted_text += f"Question: {doc.get('question', '')}\n"
             formatted_text += f"Answer: {doc.get('answer', '')}\n\n"

        return formatted_text 
    def _load_datasets(self) -> pd.DataFrame:
        """
        Load and prepare knowledge datasets
        
        Returns:
            pd.DataFrame: Combined dataset with questions, answers and metadata
        """
        try:
            # Load abortion FAQ dataset
            df_abortion = pd.read_csv("data/AbortionPPDFAQ.csv", skiprows=1)  # Skip the header row
            
            # Ensure column names are correct for the abortion dataset
            if "question" in df_abortion.columns and "answer" in df_abortion.columns:
                df_abortion = df_abortion.rename(columns={
                    "question": "Question",
                    "answer": "Answer"
                })
            
            # Handle Link column in abortion dataset
            if "Link" in df_abortion.columns and "URL" not in df_abortion.columns:
                df_abortion = df_abortion.rename(columns={"Link": "URL"})
            
            # Add source if not present
            if "Source" not in df_abortion.columns:
                df_abortion["Source"] = "Planned Parenthood Abortion FAQ"
            
            # Load general reproductive health dataset
            df_general = pd.read_csv("data/Planned Parenthood Data - Sahana.csv")
            
            # Ensure consistent column names for the general dataset
            if "question" in df_general.columns and "answer" in df_general.columns:
                df_general = df_general.rename(columns={
                    "question": "Question",
                    "answer": "Answer"
                })
            elif "Title" in df_general.columns and "Content" in df_general.columns:
                df_general = df_general.rename(columns={
                    "Title": "Question", 
                    "Content": "Answer"
                })
            
            # Handle the Link column in general dataset
            if "Link" in df_general.columns and "URL" not in df_general.columns:
                df_general = df_general.rename(columns={"Link": "URL"})
            
            # Add source if not present
            if "Source" not in df_general.columns:
                df_general["Source"] = "Planned Parenthood"
            
            # Ensure URL exists in both datasets
            if "URL" not in df_general.columns:
                df_general["URL"] = "https://www.plannedparenthood.org/learn"
            
            if "URL" not in df_abortion.columns:
                df_abortion["URL"] = "https://www.plannedparenthood.org/learn/abortion"
            
            # Combine datasets
            df_combined = pd.concat([df_abortion, df_general], ignore_index=True)
            
            # Clean data
            df_combined = df_combined.dropna(subset=["Question", "Answer"])
            df_combined["Question"] = df_combined["Question"].astype(str)
            df_combined["Answer"] = df_combined["Answer"].astype(str)
            df_combined["URL"] = df_combined["URL"].astype(str)  # Ensure URL is string
            
            logger.info(f"Loaded dataset with {len(df_combined)} entries")
            return df_combined
            
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}", exc_info=True)
            # Return an empty dataframe with the required columns
            return pd.DataFrame(columns=["Question", "Answer", "Source", "URL"])
    
    def _build_index(self) -> Tuple[faiss.Index, np.ndarray]:
        """
        Build FAISS index for efficient similarity search
        
        Returns:
            Tuple[faiss.Index, np.ndarray]: FAISS index and document embeddings
        """
        try:
            if self.embedding_model is None or len(self.data) == 0:
                logger.warning("Cannot build index: model or data not available")
                return None, np.array([])
            
            # Create a combined text field for indexing (question + answer)
            texts = self.data["Question"].tolist()
            
            # Generate embeddings
            embeddings = self._generate_embeddings(texts)
            
            if embeddings.size == 0:
                logger.warning("Generated empty embeddings, cannot build index")
                return None, np.array([])
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            logger.info(f"Built FAISS index with {len(texts)} documents and dimension {dimension}")
            return index, embeddings
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}", exc_info=True)
            return None, np.array([])
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            np.ndarray: Embeddings array
        """
        try:
            if self.embedding_model is None:
                logger.warning("Model not available for generating embeddings")
                return np.array([])
            
            embeddings = []
            batch_size = 32
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize and get embeddings
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=512)
                
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                
                # Use mean pooling to get sentence embeddings
                attention_mask = inputs['attention_mask']
                mean_embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
                
                # Normalize
                normalized_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
                embeddings.append(normalized_embeddings.numpy())
            
            return np.vstack(embeddings) if embeddings else np.array([])
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            return np.array([])
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling on token embeddings
        
        Args:
            token_embeddings: Token embeddings from BERT
            attention_mask: Attention mask for padding
            
        Returns:
            torch.Tensor: Mean-pooled embeddings
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    async def _retrieve_context(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query (str): The query to search for
            top_k (int): Number of documents to retrieve
            
        Returns:
            Tuple[List[Dict[str, Any]], List[float]]: Retrieved documents and their similarity scores
        """
        try:
            if self.index is None or len(self.data) == 0:
                logger.warning("Cannot retrieve context: index or data not available")
                return [], []
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])
            
            if query_embedding.size == 0:
                logger.warning("Generated empty query embedding, cannot search")
                return [], []
            
            # Search the index
            k = min(top_k, len(self.data))
            distances, indices = self.index.search(query_embedding, k)
            
            # Convert to document objects
            documents = []
            scores = distances[0].tolist()  # Convert to Python list
            
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.data):  # Check if index is valid
                    row = self.data.iloc[idx]
                    document = {
                        "question": row.get("Question", ""),
                        "answer": row.get("Answer", ""),
                        "source": row.get("Source", "Unknown Source"),
                    }
                    
                    # Check for URL in both URL and Link columns
                    url = row.get("URL", "")
                    if not url and "Link" in row:
                        url = row.get("Link", "")
                    document["url"] = url
                    
                    # Add similarity score
                    document["score"] = float(scores[i])
                    documents.append(document)
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents, scores
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            return [], []
    
    async def _generate_with_openai(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using OpenAI with retrieved context
        
        Args:
            query (str): The user's query
            docs (List[Dict[str, Any]]): Retrieved documents
            
        Returns:
            str: Generated response
        """
        if not self.client:
            raise ValueError("OpenAI client not available")
        
        # Prepare context from retrieved documents
        context = ""
        for i, doc in enumerate(docs):
            context += f"Document {i+1}:\n"
            context += f"Question: {doc['question']}\n"
            context += f"Answer: {doc['answer']}\n"
            context += f"Source: {doc['source']}\n\n"
        
        system_message = """You are an expert reproductive health assistant specialized in providing accurate information.
Use the provided context to answer the user's question about reproductive health.
If the context doesn't contain the necessary information to fully answer the question, acknowledge that
and provide what information you do have from the context.
Your response should be concise, accurate, and helpful. Format citations in your answer as [1], [2], etc.
IMPORTANT: Only use information from the provided context. Don't make up information or cite sources not in the context."""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",  # Use the most advanced model for best quality
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {str(e)}", exc_info=True)
            # Fall back to simple retrieval
            return self._combine_retrieved_docs(docs, [0.8] * len(docs))
    
    def _combine_retrieved_docs(self, docs: List[Dict[str, Any]], 
                               scores: List[float]) -> str:
        """
        Combine retrieved documents into a coherent response
        
        Args:
            docs (List[Dict[str, Any]]): Retrieved documents
            scores (List[float]): Similarity scores for the documents
            
        Returns:
            str: Combined response
        """
        if not docs:
            return "I don't have specific information about that. Could you ask another question about reproductive health?"
        
        # Use the most relevant document as the primary response
        primary_doc = docs[0]
        primary_answer = primary_doc['answer']
        
        # For very short responses, add information from other documents
        if len(primary_answer.split()) < 30 and len(docs) > 1:
            additional_info = []
            for i, doc in enumerate(docs[1:3]):  # Use up to 2 additional docs
                # Only add if it adds new information
                if not self._is_similar_to(doc['answer'], primary_answer, 0.7):
                    additional_info.append(doc['answer'])
            
            if additional_info:
                combined_response = primary_answer + " " + " ".join(additional_info)
                return combined_response
        
        return primary_answer
    
    def _is_similar_to(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """
        Check if two texts are similar
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            threshold (float): Similarity threshold
            
        Returns:
            bool: True if texts are similar
        """
        # Simple word overlap for similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_sim = len(intersection) / len(union)
        return jaccard_sim > threshold
    
    def _extract_citations(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract formatted citations from documents
        
        Args:
            docs (List[Dict[str, Any]]): Retrieved documents
            
        Returns:
            List[Dict[str, Any]]: Formatted citations
        """
        unique_citations = {}
        
        for i, doc in enumerate(docs):
            source = doc.get('source', 'Unknown Source')
            
            # Check for URL in both lowercase and uppercase keys
            url = doc.get('url', '')
            if not url and 'URL' in doc:
                url = doc.get('URL', '')
            
            # Create a unique key to avoid duplicate citations
            key = f"{source}_{url}"
            
            if key not in unique_citations:
                citation_obj = {
                    "id": i + 1,
                    "source": source,
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
                
                # Add the URL if available
                if url:
                    citation_obj["url"] = url
                    
                # Add the text snippet if available
                if "answer" in doc:
                    text_snippet = doc["answer"]
                    max_len = 100
                    if len(text_snippet) > max_len:
                        text_snippet = text_snippet[:max_len] + "..."
                    citation_obj["text"] = text_snippet
                
                unique_citations[key] = citation_obj
        
        return list(unique_citations.values()) 