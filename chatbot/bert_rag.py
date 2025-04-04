import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
from utils.data_loader import load_reproductive_health_data
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from chatbot.citation_manager import CitationManager, Citation

# Check for required NLTK resources - main.py should have already downloaded them
# If not, we'll download them here as a fallback
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class BertRAGModel:
    """
    Enhanced BERT-based Retrieval-Augmented Generation model for reproductive health information
    with improved vector search, hybrid retrieval, and reranking.
    """
    def __init__(self):
        """Initialize the BERT RAG model with improved embeddings and retrieval algorithms"""
        logger.info("Initializing BERT RAG Model")
        try:
            # Load pre-trained model and tokenizer
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            # Stemmer for text preprocessing
            self.stemmer = PorterStemmer()

            # Define stopwords manually to avoid NLTK dependency issues
            self.stop_words = {
                'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such',
                'when', 'while', 'who', 'whom', 'where', 'why', 'how', 'all', 'any', 'both',
                'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
                'don', 'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                'they', 'them', 'their', 'theirs', 'themselves', 'am', 'is', 'are', 'was',
                'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under',
                'again', 'further', 'for', 'of', 'by', 'about', 'against', 'between', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'with',
                'at', 'on', 'by'
            }

            # Load and index the data
            self.qa_pairs = load_reproductive_health_data()

            # Build both vector and keyword indexes
            self.build_indexes()

            # Configure additional retrieval settings
            self.synonyms = self._load_synonyms()

            logger.info("BERT RAG Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BERT RAG Model: {str(e)}", exc_info=True)
            raise

    def _load_synonyms(self):
        """Load reproductive health synonyms for query expansion"""
        # Dictionary of common synonyms and related terms for query expansion
        return {
            "abortion": ["pregnancy termination", "terminate pregnancy", "abortion care"],
            "birth control": ["contraception", "contraceptive", "birth control methods"],
            "std": ["sexually transmitted disease", "sexually transmitted infection", "sti"],
            "sti": ["sexually transmitted infection", "sexually transmitted disease", "std"],
            "morning after pill": ["plan b", "emergency contraception"],
            "iud": ["intrauterine device", "coil"],
            "menstruation": ["period", "menstrual cycle", "monthly bleeding"],
            "pregnancy": ["pregnant", "expecting", "conception"],
            "safe sex": ["protected sex", "safer sex", "condom use"],
            "rape": ["sexual assault", "sexual violence"],
        }

    def build_indexes(self):
        """Build both vector and keyword indexes for hybrid search"""
        logger.info("Building FAISS index for RAG model")
        try:
            # Extract questions and generate embeddings for vector search
            questions = [qa['Question'] for qa in self.qa_pairs]
            self.question_embeddings = self.generate_embeddings(questions)

            # Create FAISS index
            self.dimension = self.question_embeddings.shape[1]
            # Use IndexFlatIP for inner product similarity (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self.question_embeddings)

            # Store processed questions for later use
            self.raw_questions = questions
            self.processed_questions = [self._preprocess_text(q) for q in questions]

            logger.info(f"FAISS index built with {len(questions)} questions")
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}", exc_info=True)
            raise

    def _preprocess_text(self, text):
        """Preprocess text for keyword search (tokenization, lowercasing, stemming)"""
        # Remove punctuation and convert to lowercase
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)

        # Simple tokenization without relying on nltk's word_tokenize
        tokens = text.split()

        # Remove stopwords and stem
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]

        return tokens

    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts using BERT

        Args:
            texts (list): List of text strings to embed

        Returns:
            numpy.ndarray: Normalized embeddings
        """
        embeddings = []

        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize and get BERT embeddings
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use mean pooling to get sentence embeddings
            attention_mask = inputs['attention_mask']
            mean_embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)

            # Normalize
            normalized_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            embeddings.append(normalized_embeddings.numpy())

        return np.vstack(embeddings)

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

    def _is_conversational_query(self, question):
        """Check if the query is conversational rather than informational."""
        question = question.lower()
        greetings = ["hi", "hello", "hey", "greetings", "howdy", "how are you"]
        goodbyes = ["bye", "goodbye", "see you", "farewell", "exit", "quit"]

        # Check if any greeting or goodbye is in the question
        is_greeting = any(greeting in question for greeting in greetings)
        is_goodbye = any(goodbye in question for goodbye in goodbyes)

        if is_goodbye:
            return "goodbye"
        elif is_greeting:
            return "greeting"
        return False

    def _is_out_of_scope(self, question):
        """Check if question is outside the scope of reproductive health"""
        question_lower = question.lower()
        
        # Special check for ethical questions or general abortion types
        if "abortion murder" in question_lower or "abortion killing" in question_lower or "abortion moral" in question_lower or "abortion ethics" in question_lower or "types of abortion" in question_lower or "different types of abortion" in question_lower or "abortion methods" in question_lower:
            return False  # Not out of scope, use RAG first then optionally ask about policy

        # Define reproductive health related terms
        reproductive_health_terms = ["birth control", "contraception", "pregnancy", "abortion", "std", "sti", 
                                   "sexually transmitted", "period", "menstruation", "reproductive", "fertility",
                                   "sex", "sexual", "condom", "iud", "pill", "morning after", "plan b", 
                                   "reproductive health", "vagina", "penis", "uterus", "cervix", "sperm",
                                   "egg", "embryo", "fetus", "gestation", "trimester", "ovulation"]

        # Define common out-of-scope topics and their related keywords
        out_of_scope_topics = {
            "weather": ["weather", "temperature", "rain", "sunny", "cloudy", "snow", "forecast", "climate", "humidity", "degrees", "hot", "cold"],
            "politics": ["politics", "election", "government", "president", "vote", "congress", "senate", "political", "democrat", "republican", "law", "politician"],
            "technology": ["computer", "smartphone", "laptop", "tablet", "software", "app", "code", "program", "device", "internet", "website", "tech", "gadget"],
            "travel": ["travel", "flight", "hotel", "vacation", "trip", "tourism", "destination", "airfare", "airline", "beach", "resort", "passport"],
            "food": ["recipe", "cook", "food", "meal", "restaurant", "cuisine", "ingredient", "diet", "nutrition", "eat", "dinner", "lunch", "breakfast"],
            "sports": ["game", "team", "player", "score", "win", "lose", "sport", "match", "tournament", "champion", "football", "basketball", "soccer", "baseball"],
            "entertainment": ["movie", "film", "show", "series", "actor", "music", "song", "artist", "album", "concert", "tv", "television", "celebrity", "theater"],
            "emotional_expression": ["hate", "love", "angry", "mad", "upset", "sad", "depressed", "lonely", "happy", "excited", "worried", "anxious", "scared", "afraid", "hurt", "pain", "suffer", "confused", "lost", "hopeless", "helpless", "tired", "exhausted", "overwhelmed", "stressed"]
        }

        # Check if any reproductive health terms are in the question
        has_reproductive_terms = any(term in question_lower for term in reproductive_health_terms)

        # Check if any out of scope topics are in the question
        detected_topics = []
        for topic, keywords in out_of_scope_topics.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_topics.append(topic)

        # Check for general greeting or off-topic patterns
        general_query = self._is_general_query(question_lower)

        # If no reproductive health terms are present and we have detected out-of-scope topics,
        # or if the query appears to be about something else entirely
        if (not has_reproductive_terms and detected_topics) or general_query:
            return detected_topics if detected_topics else ["general"]

        return False

    def _is_general_query(self, question_lower):
        """Check if this is a general query unrelated to reproductive health"""
        # General questions and off-topic patterns
        general_patterns = [
            "what's the time", "what time is it", "what day is it", "what is your name",
            "who are you", "what can you do", "how old are you", "where are you from",
            "tell me a joke", "tell me about yourself", "what's your favorite", "what is your favorite",
            "hello world", "test", "can you help me with", "what's happening", "what is happening",
            "how's it going", "how does this work", "what's new", "what is new"
        ]

        # Check if question matches any general patterns
        for pattern in general_patterns:
            if pattern in question_lower:
                return True

        # Check if question is very short (likely not reproductive health specific)
        # and doesn't contain any health-related terms
        health_terms = ["health", "medical", "doctor", "pill", "pregnancy", "sex", "birth"]
        if len(question_lower.split()) <= 3 and not any(term in question_lower for term in health_terms):
            return True

        return False

    def expand_query(self, question):
        """
        Expand query with synonyms and related terms for better recall

        Args:
            question (str): The original user question

        Returns:
            str: Expanded question with relevant terms
        """
        question_lower = question.lower()
        expansions = []

        # Add synonyms for terms in the question
        for term, synonyms in self.synonyms.items():
            if term in question_lower:
                for synonym in synonyms:
                    # Only add if not already in the question
                    if synonym not in question_lower:
                        expansions.append(synonym)

        # Add expansions if found
        if expansions:
            expanded_question = f"{question} {' '.join(expansions)}"
            logger.debug(f"Expanded query: '{question}' -> '{expanded_question}'")
            return expanded_question

        return question

    def get_semantic_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        # Generate embeddings
        embeddings = self.generate_embeddings([text1, text2])

        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1])

        return float(similarity)

    def get_response(self, question, top_k=5):
        """
        Get response for a given question using RAG

        Args:
            question (str): The question to answer
            top_k (int): Number of similar questions to retrieve

        Returns:
            str: The answer to the question
        """
        try:
            # Import citation manager
            citation_mgr = CitationManager()

            # Check if this is a conversational query instead of a health question
            conversational_type = self._is_conversational_query(question)
            if conversational_type == "greeting":
                logger.debug(f"Detected conversational query: '{question}'")
                greeting_response = "I'm doing well, thanks for asking! How can I help you with reproductive health information today?"
                return citation_mgr.add_citation_to_text(greeting_response, "planned_parenthood")
            elif conversational_type == "goodbye":
                goodbye_response = "Goodbye! Take care and stay healthy."
                return citation_mgr.add_citation_to_text(goodbye_response, "planned_parenthood")

            # Check if the question is out of scope
            out_of_scope = self._is_out_of_scope(question)
            if out_of_scope:
                topics = ", ".join(out_of_scope)
                logger.debug(f"Detected out-of-scope query about {topics}: '{question}'")
                out_of_scope_response = self._get_out_of_scope_response(out_of_scope)
                return citation_mgr.add_citation_to_text(out_of_scope_response, "planned_parenthood")

            # First check for exact matches (case-insensitive) to prioritize them
            normalized_question = question.lower().strip('?. ')

            # Check for exact match
            for idx, qa_pair in enumerate(self.qa_pairs):
                qa_normalized = qa_pair['Question'].lower().strip('?. ')
                # Check if this is an exact match
                if normalized_question == qa_normalized:
                    logger.debug(f"Found exact match for question: '{question}'")
                    logger.debug(f"Exact match index: {idx}")
                    answer = qa_pair['Answer']
                    # Use the link from the QA pair if available
                    link = qa_pair.get('Link', '')
                    if link and len(link) > 0:
                        # Create a custom citation for this specific link
                        citation_mgr.sources["custom_link"] = Citation(
                            source="Planned Parenthood",
                            url=link,
                            title="Planned Parenthood",
                            authors=["Planned Parenthood Federation of America"]
                        )
                        return citation_mgr.add_citation_to_text(answer, "custom_link")
                    else:
                        return citation_mgr.add_citation_to_text(answer, "planned_parenthood")

            # Also check for questions that contain the exact query
            # This helps with cases like "what is the menstrual cycle" matching "what is the menstrual cycle?"
            for idx, qa_pair in enumerate(self.qa_pairs):
                qa_normalized = qa_pair['Question'].lower().strip('?. ')
                if qa_normalized.startswith(normalized_question) or normalized_question.startswith(qa_normalized):
                    logger.debug(f"Found partial match for question: '{question}'")
                    logger.debug(f"Partial match index: {idx}")
                    answer = qa_pair['Answer']
                    # Use the link from the QA pair if available
                    link = qa_pair.get('Link', '')
                    if link and len(link) > 0:
                        # Create a custom citation for this specific link
                        citation_mgr.sources["custom_link"] = Citation(
                            source="Planned Parenthood",
                            url=link,
                            title="Planned Parenthood",
                            authors=["Planned Parenthood Federation of America"]
                        )
                        return citation_mgr.add_citation_to_text(answer, "custom_link")
                    else:
                        return citation_mgr.add_citation_to_text(answer, "planned_parenthood")

            # Expand the query for better recall
            expanded_question = self.expand_query(question)

            # If no exact match, proceed with embedding-based retrieval
            # Generate embedding for the question
            question_embedding = self.generate_embeddings([expanded_question])

            # Search for similar questions
            distances, indices = self.index.search(question_embedding, top_k)

            # Perform a confidence check - don't answer if distance is too high
            if distances[0][0] > 15.0:
                logger.debug(f"Low confidence (distance: {distances[0][0]}) for query: '{question}'")
                response = "I'm not sure I understand your question about reproductive health. Could you please rephrase it or ask something more specific about contraception, pregnancy, or reproductive health?"
                # Add Planned Parenthood citation by default even for uncertainty responses
                return citation_mgr.add_citation_to_text(response, "planned_parenthood")

            # If multiple good matches, combine answers
            if len(indices[0]) > 1 and distances[0][1] < 12.0:
                return self._combine_top_answers(question, distances[0], indices[0])

            # Get the most similar question's answer
            best_idx = indices[0][0]
            best_answer = self.qa_pairs[best_idx]['Answer']
            best_question = self.qa_pairs[best_idx]['Question']

            # Add citation with the link from the QA pair if available
            link = self.qa_pairs[best_idx].get('Link', '')
            if link and len(link) > 0:
                # Create a custom citation for this specific link
                citation_mgr.sources["custom_link"] = Citation(
                    source="Planned Parenthood",
                    url=link,
                    title="Planned Parenthood",
                    authors=["Planned Parenthood Federation of America"]
                )
                cited_answer = citation_mgr.add_citation_to_text(best_answer, "custom_link")
            else:
                cited_answer = citation_mgr.add_citation_to_text(best_answer, "planned_parenthood")

            logger.debug(f"Primary confidence (distance): {distances[0][0]}")
            if len(indices[0]) > 1:
                logger.debug(f"Secondary confidence (gap): {distances[0][1] - distances[0][0]}")
            logger.debug(f"Matched question: {best_question}")

            # Return the answer with citation
            return cited_answer

        except Exception as e:
            logger.error(f"Error getting RAG response: {str(e)}", exc_info=True)
            error_response = "I apologize, but I encountered an error processing your question. Please try asking again or rephrase your question."

            # Add citation even for error responses
            try:
                return citation_mgr.add_citation_to_text(error_response, "planned_parenthood")
            except:
                # If citation fails, return plain error message
                return error_response
                
    def get_response_with_context(self, question, top_k=5):
        """
        Get response with context information for a given question using RAG

        Args:
            question (str): The question to answer
            top_k (int): Number of similar questions to retrieve

        Returns:
            tuple: (answer string, context list)
        """
        try:
            # Import citation manager
            citation_mgr = CitationManager()
            
            # Prepare empty contexts list
            contexts = []

            # Check if this is a conversational query instead of a health question
            conversational_type = self._is_conversational_query(question)
            if conversational_type == "greeting":
                logger.debug(f"Detected conversational query: '{question}'")
                greeting_response = "I'm doing well, thanks for asking! How can I help you with reproductive health information today?"
                return citation_mgr.add_citation_to_text(greeting_response, "planned_parenthood"), contexts
            elif conversational_type == "goodbye":
                goodbye_response = "Goodbye! Take care and stay healthy."
                return citation_mgr.add_citation_to_text(goodbye_response, "planned_parenthood"), contexts

            # Check if the question is out of scope
            out_of_scope = self._is_out_of_scope(question)
            if out_of_scope:
                topics = ", ".join(out_of_scope)
                logger.debug(f"Detected out-of-scope query about {topics}: '{question}'")
                out_of_scope_response = self._get_out_of_scope_response(out_of_scope)
                return citation_mgr.add_citation_to_text(out_of_scope_response, "planned_parenthood"), contexts

            # First check for exact matches (case-insensitive) to prioritize them
            normalized_question = question.lower().strip('?. ')

            # Check for exact match
            for idx, qa_pair in enumerate(self.qa_pairs):
                qa_normalized = qa_pair['Question'].lower().strip('?. ')
                # Check if this is an exact match
                if normalized_question == qa_normalized:
                    logger.debug(f"Found exact match for question: '{question}'")
                    logger.debug(f"Exact match index: {idx}")
                    answer = qa_pair['Answer']
                    # Use the link from the QA pair if available
                    link = qa_pair.get('Link', '')
                    if link and len(link) > 0:
                        # Create a custom citation for this specific link
                        citation_mgr.sources["custom_link"] = Citation(
                            source="Planned Parenthood",
                            url=link,
                            title="Planned Parenthood",
                            authors=["Planned Parenthood Federation of America"]
                        )
                        return citation_mgr.add_citation_to_text(answer, "custom_link"), contexts
                    else:
                        contexts.append({
                            'question': qa_pair['Question'],
                            'answer': qa_pair['Answer'],
                            'distance': 0.0,  # Perfect match
                            'category': qa_pair.get('Category', 'General'),
                            'source': 'Planned Parenthood'
                        })
                        return citation_mgr.add_citation_to_text(answer, "planned_parenthood"), contexts

            # Also check for questions that contain the exact query
            # This helps with cases like "what is the menstrual cycle" matching "what is the menstrual cycle?"
            for idx, qa_pair in enumerate(self.qa_pairs):
                qa_normalized = qa_pair['Question'].lower().strip('?. ')
                if qa_normalized.startswith(normalized_question) or normalized_question.startswith(qa_normalized):
                    logger.debug(f"Found partial match for question: '{question}'")
                    logger.debug(f"Partial match index: {idx}")
                    answer = qa_pair['Answer']
                    # Use the link from the QA pair if available
                    link = qa_pair.get('Link', '')
                    if link and len(link) > 0:
                        # Create a custom citation for this specific link
                        citation_mgr.sources["custom_link"] = Citation(
                            source="Planned Parenthood",
                            url=link,
                            title="Planned Parenthood",
                            authors=["Planned Parenthood Federation of America"]
                        )
                        return citation_mgr.add_citation_to_text(answer, "custom_link"), contexts
                    else:
                        contexts.append({
                            'question': qa_pair['Question'],
                            'answer': qa_pair['Answer'],
                            'distance': 1.0,  # Very close match
                            'category': qa_pair.get('Category', 'General'),
                            'source': 'Planned Parenthood'
                        })
                        return citation_mgr.add_citation_to_text(answer, "planned_parenthood"), contexts

            # Expand the query for better recall
            expanded_question = self.expand_query(question)

            # If no exact match, proceed with embedding-based retrieval
            # Generate embedding for the question
            question_embedding = self.generate_embeddings([expanded_question])

            # Search for similar questions
            distances, indices = self.index.search(question_embedding, top_k)
            
            # Create context list from the retrieved documents
            for i in range(min(top_k, len(indices[0]))):
                idx = indices[0][i]
                dist = float(distances[0][i])
                contexts.append({
                    'question': self.qa_pairs[idx]['Question'],
                    'answer': self.qa_pairs[idx]['Answer'],
                    'distance': dist,
                    'category': self.qa_pairs[idx].get('Category', 'General'),
                    'source': 'Planned Parenthood',
                    'link': self.qa_pairs[idx].get('Link', '')
                })

            # Perform a confidence check - don't answer if distance is too high
            if distances[0][0] > 15.0:
                logger.debug(f"Low confidence (distance: {distances[0][0]}) for query: '{question}'")
                response = "I'm not sure I understand your question about reproductive health. Could you please rephrase it or ask something more specific about contraception, pregnancy, or reproductive health?"
                # Add Planned Parenthood citation by default even for uncertainty responses
                return citation_mgr.add_citation_to_text(response, "planned_parenthood"), contexts

            # If multiple good matches, combine answers
            if len(indices[0]) > 1 and distances[0][1] < 12.0:
                combined = self._combine_top_answers(question, distances[0], indices[0])
                return combined, contexts

            # Get the most similar question's answer
            best_idx = indices[0][0]
            best_answer = self.qa_pairs[best_idx]['Answer']
            best_question = self.qa_pairs[best_idx]['Question']

            # Add citation with the link from the QA pair if available
            link = self.qa_pairs[best_idx].get('Link', '')
            if link and len(link) > 0:
                # Create a custom citation for this specific link
                citation_mgr.sources["custom_link"] = Citation(
                    source="Planned Parenthood",
                    url=link,
                    title="Planned Parenthood",
                    authors=["Planned Parenthood Federation of America"]
                )
                cited_answer = citation_mgr.add_citation_to_text(best_answer, "custom_link")
            else:
                cited_answer = citation_mgr.add_citation_to_text(best_answer, "planned_parenthood")

            logger.debug(f"Primary confidence (distance): {distances[0][0]}")
            if len(indices[0]) > 1:
                logger.debug(f"Secondary confidence (gap): {distances[0][1] - distances[0][0]}")
            logger.debug(f"Matched question: {best_question}")

            # Return the answer with citation
            return cited_answer, contexts

        except Exception as e:
            logger.error(f"Error getting RAG response with context: {str(e)}", exc_info=True)
            error_response = "I apologize, but I encountered an error processing your question. Please try asking again or rephrase your question."

            # Add citation even for error responses
            try:
                return citation_mgr.add_citation_to_text(error_response, "planned_parenthood"), []
            except:
                # If citation fails, return plain error message
                return error_response, []

    def _combine_top_answers(self, question, distances, indices, max_answers=3):
        """
        Combine the top answers for better response with natural language flow

        Args:
            question (str): The original question
            distances (list): Distances of top matches
            indices (list): Indices of top matches
            max_answers (int): Maximum number of answers to combine

        Returns:
            str: Combined answer with improved natural language flow
        """
        relevant_answers = []

        # Get the top answers that are within a reasonable distance
        for i in range(min(max_answers, len(indices))):
            if distances[i] < 15.0:  # Only include answers with reasonable similarity
                idx = indices[i]
                answer = self.qa_pairs[idx]['Answer']
                question_match = self.qa_pairs[idx]['Question']
                category = self.qa_pairs[idx].get('Category', 'General')
                link = self.qa_pairs[idx].get('Link', '')

                relevant_answers.append({
                    'answer': answer,
                    'question': question_match,
                    'distance': distances[i],
                    'category': category,
                    'link': link
                })

        # Create citation manager if not already imported at the class level
        citation_mgr = CitationManager()

        # If only one good match, format it for better readability
        if len(relevant_answers) == 1:
            answer = self._format_single_answer(question, relevant_answers[0]['answer'])
            
            # Use the link from the QA pair if available
            link = relevant_answers[0]['link']
            if link and len(link) > 0:
                # Create a custom citation for this specific link
                citation_mgr.sources["custom_link"] = Citation(
                    source="Planned Parenthood",
                    url=link,
                    title="Planned Parenthood",
                    authors=["Planned Parenthood Federation of America"]
                )
                return citation_mgr.add_citation_to_text(answer, "custom_link")
            else:
                return citation_mgr.add_citation_to_text(answer, "planned_parenthood")

        # Combine multiple answers into a comprehensive, naturally flowing response
        return self._format_multiple_answers(question, relevant_answers)

    def _format_single_answer(self, question, answer):
        """
        Format a single answer to be concise and direct

        Args:
            question (str): Original question
            answer (str): Answer to format

        Returns:
            str: Formatted answer
        """
        # Make sure the answer ends with proper punctuation
        if not answer.endswith(('.', '?', '!')):
            answer = answer + '.'

        # For long answers, extract the most important parts (first 2 sentences)
        sentences = self._extract_sentences(answer)

        if len(sentences) > 2:
            # Use first 2 sentences for concise response
            concise_answer = ' '.join(sentences[:2])

            return concise_answer

        return answer

    def _format_multiple_answers(self, question, relevant_answers):
        """
        Format multiple answers into a cohesive, concise response

        Args:
            question (str): Original question
            relevant_answers (list): List of relevant answer dictionaries

        Returns:
            str: Formatted and combined answer
        """
        # Create citation manager
        citation_mgr = CitationManager()

        # Sort answers by relevance (distance)
        sorted_answers = sorted(relevant_answers, key=lambda x: x['distance'])

        # Extract the most relevant answer as our primary response
        primary_answer = sorted_answers[0]['answer']
        primary_link = sorted_answers[0].get('link', '')

        # Find one important sentence from other answers that adds new info
        important_point = ""
        seen_content = set(self._get_key_phrases(primary_answer))

        for item in sorted_answers[1:]:
            sentences = self._extract_sentences(item['answer'])
            if not sentences:
                continue

            # Get the first sentence that adds new information
            for sentence in sentences[:1]:  # Only look at first sentence
                key_phrases = self._get_key_phrases(sentence)
                # Check if this adds new information
                if not any(phrase in seen_content for phrase in key_phrases):
                    important_point = sentence
                    break

            if important_point:
                break

        # Include more content from the primary answer (3 sentences instead of 1)
        direct_answer = self._get_first_sentences(primary_answer, 3)

        # Build a more comprehensive response by including more information
        # Add the full primary answer rather than just the first sentence
        if important_point:
            if not important_point.strip().endswith(('.', '?', '!')):
                important_point = important_point + '.'
            comprehensive_response = primary_answer + ' ' + important_point
        else:
            comprehensive_response = primary_answer

        # Add citation with the primary answer's link if available
        if primary_link and len(primary_link) > 0:
            # Create a custom citation for this specific link
            citation_mgr.sources["custom_link"] = Citation(
                source="Planned Parenthood",
                url=primary_link,
                title="Planned Parenthood",
                authors=["Planned Parenthood Federation of America"]
            )
            cited_response = citation_mgr.add_citation_to_text(comprehensive_response, "custom_link") 
        else:
            cited_response = citation_mgr.add_citation_to_text(comprehensive_response, "planned_parenthood") 
        
        return cited_response

    def _get_first_sentences(self, text, num_sentences=2):
        """Extract the first N sentences from a text"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return ' '.join(sentences[:num_sentences])

    def _extract_sentences(self, text):
        """Split text into sentences"""
        import re
        return re.split(r'(?<=[.!?])\s+', text)

    def _get_key_phrases(self, text):
        """Extract key phrases/words from text to identify content"""
        # Simple implementation: just use words of 4+ chars
        words = [w.lower() for w in text.split() if len(w) >= 4]
        return set(words)

    def is_confident(self, question, response, threshold=6.0):
        """
        Determine if the RAG model is confident in its response
        Uses multiple metrics for better confidence assessment

        Args:
            question (str): The original question
            response (str): The generated response
            threshold (float): Confidence threshold (lower is more confident)

        Returns:
            bool: True if confident, False otherwise
        """
        try:
            # Check if the question is out of scope
            if self._is_out_of_scope(question):
                logger.debug("Question is out of scope, not confident")
                return False

            # Generate embedding for the question
            question_embedding = self.generate_embeddings([question])

            # Search for the closest questions
            distances, indices = self.index.search(question_embedding, 3)

            # Primary confidence: distance to closest match
            primary_confidence = distances[0][0]

            # Secondary confidence: gap between first and second match
            # A big gap means the first match is distinctly better
            secondary_confidence = 0
            if len(distances[0]) > 1:
                secondary_confidence = distances[0][1] - distances[0][0]

            # Get the matched question text for semantic similarity check
            matched_question = self.qa_pairs[indices[0][0]]['Question']

            # Log confidence metrics
            logger.debug(f"Primary confidence (distance): {primary_confidence}")
            logger.debug(f"Secondary confidence (gap): {secondary_confidence}")
            logger.debug(f"Matched question: {matched_question}")

            # Strict confidence checks to avoid incorrect information

            # 1. If primary confidence is very good, trust it
            if primary_confidence < threshold * 0.6:
                logger.debug("High confidence based on primary score")
                return True

            # 2. If primary is good and secondary shows a clear winner, trust it
            if primary_confidence < threshold * 0.8 and secondary_confidence > 2.5:
                logger.debug("Confidence based on primary score and distinct winner")
                return True

            # 3. If primary is just above threshold but very close to original question semantically, trust it
            if primary_confidence < threshold and self._is_semantically_similar(question, matched_question):
                similarity_score = self._get_semantic_similarity_score(question, matched_question)
                logger.debug(f"Semantic similarity score: {similarity_score}")
                if similarity_score > 0.6:  # Require higher similarity
                    logger.debug("Confidence based on high semantic similarity")
                    return True

            # Otherwise, not confident - be conservative to avoid misinformation
            logger.debug("Not confident enough to provide an answer")
            return False

        except Exception as e:
            logger.error(f"Error checking confidence: {str(e)}", exc_info=True)
            return False

    def _is_semantically_similar(self, q1, q2):
        """
        Check if two questions are semantically similar

        Args:
            q1 (str): First question
            q2 (str): Second question

        Returns:
            bool: True if semantically similar
        """
        similarity = self._get_semantic_similarity_score(q1, q2)
        logger.debug(f"Semantic similarity: {similarity}")
        return similarity > 0.4

    def _get_semantic_similarity_score(self, q1, q2):
        """
        Calculate semantic similarity score between two questions

        Args:
            q1 (str): First question
            q2 (str): Second question

        Returns:
            float: Similarity score between 0 and 1
        """
        # Simple keyword matching with improved preprocessing
        # Remove common stop words and punctuation
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'of', 'from'}

        def preprocess(text):
            # Convert to lowercase and remove punctuation
            text = text.lower()
            text = ''.join([c for c in text if c.isalnum() or c.isspace()])
            # Split into words and remove stop words
            words = [w for w in text.split() if w not in stop_words]
            return set(words)

        q1_words = preprocess(q1)
        q2_words = preprocess(q2)

        # Calculate Jaccard similarity
        intersection = len(q1_words.intersection(q2_words))
        union = len(q1_words.union(q2_words))

        if union == 0:
            return 0.0

        # Basic Jaccard similarity
        basic_similarity = intersection / union

        # Check for exact phrase matches (more weight for exact matches)
        q1_phrases = [' '.join(q1.lower().split()[i:i+3]) for i in range(len(q1.lower().split())-2)]
        q2_phrases = [' '.join(q2.lower().split()[i:i+3]) for i in range(len(q2.lower().split())-2)]

        phrase_matches = sum(1 for p in q1_phrases if p in q2_phrases)
        phrase_similarity = phrase_matches / max(len(q1_phrases), len(q2_phrases), 1) if q1_phrases and q2_phrases else 0

        # Combined similarity (weighted)
        similarity = (basic_similarity * 0.7) + (phrase_similarity * 0.3)

        return similarity

    def _get_out_of_scope_response(self, topics):
        """Generate a response for out-of-scope questions"""
        # Default response for out-of-scope topics
        primary_topic = topics[0] if topics else "general"

        topic_responses = {
            "weather": "I'm designed to provide information about reproductive health, not weather forecasts. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "politics": "I'm here to provide information about reproductive health, not political matters. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "technology": "I'm programmed to assist with reproductive health questions, not technology matters. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "travel": "I'm designed to answer questions about reproductive health, not travel information. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "food": "I'm trained to provide information about reproductive health, not food or nutrition in general. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "sports": "I'm programmed to assist with reproductive health questions, not sports information. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "entertainment": "I'm designed to provide information about reproductive health, not entertainment. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "emotional_expression": "I understand you're expressing a personal feeling. While I'm here to listen, I'm specifically designed to provide information about reproductive health topics. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those.",
            "general": "I'm Abby, a chatbot specifically designed to provide information about reproductive health. I don't have information on this topic. If you have any questions about contraception, pregnancy, abortion access, or sexual health, I'd be happy to help with those instead."
        }

        # Check if this is emotional expression - if so, we'll return the response WITHOUT citation
        if primary_topic == "emotional_expression":
            # Return without citation by setting a special flag that the caller should check
            return topic_responses["emotional_expression"] + "NO_CITATION"
        
        return topic_responses.get(primary_topic, topic_responses["general"])