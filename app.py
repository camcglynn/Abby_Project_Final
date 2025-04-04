import logging
import os
import uuid
import time
import json
from typing import Dict, List, Any, Optional
import asyncio
import dotenv
from datetime import datetime
import re
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request, Depends, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from utils.metrics import (
    increment_counter, record_time, record_api_call, flush_metrics,
    record_feedback,record_safety_score, record_empathy_score, 
    record_memory_usage, record_measurement
)
# In app.py, after other utils imports
try:
    from utils.response_evaluation import check_response_safety, calculate_empathy
    EVALUATION_FUNCTIONS_LOADED = True
except ImportError:
    logger.error("Could not import evaluation functions from utils.response_evaluation. Placeholders will be used.")
    EVALUATION_FUNCTIONS_LOADED = False

    # Define placeholders directly in app.py if import fails
    def check_response_safety(text: str) -> bool:
        logger.warning("Using placeholder safety check (always True)")
        return True

    def calculate_empathy(text: str) -> float:
        logger.warning("Using placeholder empathy calculation (always 0.5)")
        return 0.5
    
from datetime import datetime, timedelta # Import datetime stuff
from fastapi import APIRouter, Depends, HTTPException, Query # If using router, else adapt

# Set up environment variables
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our classes
from chatbot.multi_aspect_processor import MultiAspectQueryProcessor
from chatbot.memory_manager import MemoryManager

# Create FastAPI app
app = FastAPI(title="Abby Chatbot API", 
              description="Reproductive health chatbot with multi-aspect query handling",
              version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Initialize global components
memory_manager = MemoryManager()
# Check for serialized models
serialized_models_path = Path('serialized_models')
if serialized_models_path.exists() and os.path.isfile('load_serialized_models.py'):
    logger.info('Serialized models found, attempting to load...')
    try:
        import load_serialized_models
        if load_serialized_models.check_serialized_models():
            logger.info('Using serialized models for faster initialization')
            # Load serialized models into global components
            loaded_models = load_serialized_models.load_all_models()
            # Memory manager and query processor will still be initialized normally
        else:
            logger.info('Serialized models check failed, using normal initialization')
    except Exception as e:
        logger.error(f'Error loading serialized models: {str(e)}')
        logger.info('Falling back to normal initialization')
else:
    logger.info('No serialized models found, using normal initialization')

query_processor = MultiAspectQueryProcessor()

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_location: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    text: str
    message_id: str
    session_id: str
    citations: List[Any] = []
    citation_objects: List[Any] = []
    timestamp: float
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    graphics: Optional[List[Dict[str, Any]]] = None
    show_map: Optional[bool] = None
    zip_code: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: str

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str

class FeedbackRequest(BaseModel):
    message_id: str
    session_id: Optional[str] = None # Add if needed and sent from frontend
    rating: int
    comment: Optional[str] = None
    quality_metrics: Optional[Dict[str, float]] = None # Keep if you plan to send these

# Dependency to get the processor
def get_processor():
    return query_processor

# Dependency to get the memory manager
def get_memory_manager():
    return memory_manager

# Main page route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the main chat interface
    """
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "google_maps_api_key": os.getenv("GOOGLE_MAPS_API_KEY", "")
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, 
               processor: MultiAspectQueryProcessor = Depends(get_processor),
               memory: MemoryManager = Depends(get_memory_manager)):
    """
    Process a chat message and return a response
    """
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4()) # Ensure session_id is defined early

    try:
        logger.info(f"Received chat request: {request.message[:100]}")
        
        # Track message received
        increment_counter('user_messages')

        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())
        # Optional: Log message length as a separate event if useful
        record_measurement('user_message_length', len(request.message), session_id=session_id)

        # New session start event
        if not request.session_id:
            increment_counter('session_starts', session_id=session_id) # Log session_id with start

        # Store user message in memory
        memory.add_message(
            session_id=session_id,
            message=request.message,
            role="user",
            metadata=request.metadata
        )
        
        # Get conversation history
        conversation_history = memory.get_history(session_id)
        
        # Process the query through our multi-aspect processor
        query_start_time = time.time()
        response_data = await processor.process_query(
            message=request.message,
            conversation_history=conversation_history,
            user_location=request.user_location
        )
        query_processing_time = time.time() - query_start_time
        
        # Record query processing time
        record_time('query_processing', query_processing_time, session_id=session_id, category=response_data.get("aspect_type"))

        # Add message ID and session ID to response
        message_id = str(uuid.uuid4())
        response_data["message_id"] = message_id
        response_data["session_id"] = session_id

        if "processing_time" not in response_data:
            response_data["processing_time"] = time.time() - start_time
            
        # Fix response text if it's missing
        if "text" not in response_data and "primary_content" in response_data:
            logger.info("Using primary_content as text in response")
            response_data["text"] = response_data["primary_content"]
        
        # Clean up response text and ensure citation_objects have proper format
        if "text" in response_data:
            # Make sure text is not empty or just a period
            if not response_data["text"] or response_data["text"].strip() == ".":
                logger.warning("Response text is empty or just a period, using fallback")
                # Check if we have policy data we can use
                if "primary_content" in response_data:
                    response_data["text"] = response_data["primary_content"]
                elif response_data.get("aspect_type") == "policy" and "state_name" in response_data:
                    state_name = response_data["state_name"]
                    response_data["text"] = f"Here's abortion policy information for {state_name}. Please check official sources like Planned Parenthood for the most up-to-date information."
                else:
                    response_data["text"] = "I apologize, but I couldn't generate a complete response to your question. Could you try rephrasing or asking another question about reproductive health?"
            
            # Remove inline citations in text
            
            # Remove numbered citations [1], [2], etc. and the actual bracket character [.
            response_data["text"] = re.sub(r'\[\d+\]', '', response_data["text"])
            response_data["text"] = re.sub(r'\[\.\s*', '', response_data["text"])
            
            # Remove citations in parentheses like (Planned Parenthood, SOURCE...)
            response_data["text"] = re.sub(r'\s?\([^)]*(?:SOURCE|source)[^)]*\)', '', response_data["text"])
            
            # Remove "SOURCE" text
            response_data["text"] = re.sub(r'\s?SOURCE.+?(?=\s|$|\.|,)', '', response_data["text"])
            
            # Remove stray brackets that might be left from citation formatting
            response_data["text"] = re.sub(r'\s?\[\.?\]', '', response_data["text"])
            
            # Remove "For more information, see sources" at the end
            response_data["text"] = re.sub(
                r"For more (?:detailed )?information,?\s*(?:you can )?(?:refer to|see|check) (?:the )?(?:resources|sources)(?:\s*from [^.]+)?\.?\s*$", 
                "", 
                response_data["text"]
            )
            # Remove citation markers at the end
            response_data["text"] = re.sub(
                r"(?:\s*\[\d+\])+\s*\.?$", 
                ".", 
                response_data["text"]
            )
        
        # Extract state information from aspect_responses if available
        if "aspect_responses" in response_data:
            for aspect_response in response_data["aspect_responses"]:
                if aspect_response.get("aspect_type") == "policy":
                    # Copy state_code if it exists in this aspect response
                    if "state_code" in aspect_response and "state_code" not in response_data:
                        response_data["state_code"] = aspect_response["state_code"]
                        logger.info(f"Added state_code to response: {aspect_response['state_code']}")
                    
                    # Copy state_codes if it exists in this aspect response
                    if "state_codes" in aspect_response and "state_codes" not in response_data:
                        response_data["state_codes"] = aspect_response["state_codes"]
                        logger.info(f"Added state_codes to response: {aspect_response['state_codes']}")
                    
                    break
        
        # Log information about citations
        logger.info(f"Response has {len(response_data.get('citations', []))} citations")
        citation_objects = response_data.get('citation_objects', [])
        
        # Fix citation_objects if they're still strings
        if citation_objects and all(isinstance(c, str) for c in citation_objects):
            logger.info("Converting citation_objects from strings to objects")
            new_citation_objects = []
            
            # Check if we have aspect_responses with citation_objects that contain URLs
            aspect_citation_urls = {}
            if "aspect_responses" in response_data:
                for aspect_response in response_data["aspect_responses"]:
                    if aspect_response.get("aspect_type") == "knowledge":
                        aspect_citations = aspect_response.get("citation_objects", [])
                        if aspect_citations and isinstance(aspect_citations[0], dict):
                            # Extract URLs from the knowledge handler
                            for idx, citation in enumerate(aspect_citations):
                                source = citation.get("source")
                                url = citation.get("url")
                                title = citation.get("title")
                                if source and url:
                                    aspect_citation_urls[source + str(idx)] = {
                                        "url": url,
                                        "title": title or source
                                    }
            
            # Create new citation objects with proper URLs if available
            for idx, source in enumerate(citation_objects):
                source_key = source + str(idx)
                # Default values
                url = "https://www.plannedparenthood.org"
                title = source
                
                # If we have a specific URL from the knowledge handler, use it
                if source_key in aspect_citation_urls:
                    url = aspect_citation_urls[source_key]["url"]
                    title = aspect_citation_urls[source_key]["title"]
                
                citation_dict = {
                    "source": source,
                    "title": title,
                    "url": url,
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
                logger.info(f"Created citation object: {json.dumps(citation_dict)}")
                new_citation_objects.append(citation_dict)
            
            response_data["citation_objects"] = new_citation_objects
        
        # Remove duplicate citation objects
        if response_data.get("citation_objects"):
            # Track unique URLs
            unique_urls = {}
            unique_citation_objects = []
            
            for citation in response_data["citation_objects"]:
                if isinstance(citation, dict):
                    url = citation.get("url")
                    if url and url not in unique_urls:
                        unique_urls[url] = True
                        unique_citation_objects.append(citation)
                    # Keep non-URL citations
                    elif not url:
                        unique_citation_objects.append(citation)
                else:
                    # Keep string citations
                    unique_citation_objects.append(citation)
            
            # Update response with deduplicated citations
            response_data["citation_objects"] = unique_citation_objects
            # Update citations list too for consistency
            if "citations" in response_data:
                response_data["citations"] = [c.get("source") for c in unique_citation_objects if isinstance(c, dict) and "source" in c]
            
            logger.info(f"Removed duplicate citations, {len(response_data['citation_objects'])} unique citations remain")
            
        # Log the final citation objects
        logger.info(f"Citation objects: {json.dumps(response_data.get('citation_objects', []))}")
        
        # Store bot response in memory
        memory.add_message(
            session_id=session_id,
            message=response_data["text"],
            role="assistant",
            metadata={
                "message_id": message_id,
                "citations": response_data.get("citations", []),
                "citation_objects": response_data.get("citation_objects", [])
            }
        )
        
        # Add empty graphics array for compatibility with frontend
        if "graphics" not in response_data:
            response_data["graphics"] = []
            
        # Extract state information from aspect_responses if available
        if "aspect_responses" in response_data:
            for aspect_response in response_data["aspect_responses"]:
                if aspect_response.get("aspect_type") == "policy":
                    # Copy state_code if it exists in this aspect response
                    if "state_code" in aspect_response and "state_code" not in response_data:
                        response_data["state_code"] = aspect_response["state_code"]
                        logger.info(f"Added state_code to response: {aspect_response['state_code']}")
                    
                    # Copy state_codes if it exists in this aspect response
                    if "state_codes" in aspect_response and "state_codes" not in response_data:
                        response_data["state_codes"] = aspect_response["state_codes"]
                        logger.info(f"Added state_codes to response: {aspect_response['state_codes']}")
                    
                    break
        # Record response metrics
        record_time(
            metric_name='query_processing',
            elapsed_time=query_processing_time,
            session_id=session_id,
            category=response_data.get("aspect_type")) # Use .get() - sends None if key missing)
        
        # --- Record Metrics for the Response ---
        # Estimate token count and record API usage event
        response_text_length = len(response_data.get("text", ""))
        estimated_tokens = int(response_text_length / 4)
        record_api_call('chatbot_response', tokens_used=estimated_tokens, session_id=session_id)

        # Safety Score event (calls the imported or placeholder function)
        text_to_check = response_data.get("text", "") # Get text safely
        is_safe = check_response_safety(text_to_check)
        safety_score = 1.0 if is_safe else 0.0
        record_safety_score(safety_score, session_id=session_id, message_id=message_id)

        # Empathy Score event (calls the imported or placeholder function)
        empathy_score = calculate_empathy(text_to_check)
        record_empathy_score(empathy_score, session_id=session_id, message_id=message_id)

        # Memory Usage event
        record_memory_usage(session_id=session_id)
        # --- End Response Metrics ---

        # Store bot response in memory
        memory.add_message(
            session_id=session_id,
            message=response_data["text"],
            role="assistant",
            metadata={
                "message_id": message_id,
                "citations": response_data.get("citations", []),
                "citation_objects": response_data.get("citation_objects", [])
            }
        )

        # Track total conversation time
        total_time = time.time() - start_time
        record_time('total_response_time', total_time, session_id = session_id)
        
        # Flush metrics to ensure they're written to storage
        flush_metrics()

        return response_data
        
    except Exception as e:
        # Record error
        increment_counter('errors', session_id=session_id)
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.delete("/session", status_code=200)
async def clear_session(request: SessionRequest, memory: MemoryManager = Depends(get_memory_manager)):
    """
    Clear a conversation session
    """
    try:
        result = memory.clear_session(request.session_id)
        if result:
            return {"status": "success", "message": f"Session {request.session_id} cleared"}
        else:
            return {"status": "warning", "message": f"Session {request.session_id} not found"}
            
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@app.get("/history/{session_id}", status_code=200)
async def get_history(session_id: str, memory: MemoryManager = Depends(get_memory_manager)):
    """
    Get conversation history for a session
    """
    try:
        history = memory.get_history(session_id)
        return {"session_id": session_id, "messages": history}
            
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")

@app.post("/feedback", status_code=200)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a message
    """
    try:
        session_id = request.session_id
        message_id = request.message_id
        rating = request.rating
        comment = request.comment

        logger.info(f"Session {session_id or 'Unknown'}: Received feedback for message {message_id}: rating={rating}")

        # Record feedback in metrics
        is_positive = rating >= 3  # Assuming rating scale is 1-5
        record_feedback(
            positive=is_positive,
            message_id=message_id,
            session_id=session_id,
            rating=rating, # Pass the actual rating
            comment=comment
        )

        # --- Removed previous commented-out code related to quality_metrics/ROUGE ---

        # In a production system, store this feedback in a database
        # For now, just log it and append to a simple JSON file
        feedback_data = {
            "message_id": message_id,
            "session_id": session_id, # Added session_id here too
            "rating": rating,
            "comment": comment,
            "timestamp": time.time() # Use time.time() for consistency, or datetime.now().isoformat()
        }

        # Append to a simple JSON file
        feedback_file = os.path.join(os.getcwd(), "user_feedback.json")
        try:
            all_feedback = [] # Initialize default
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    try:
                        content = f.read()
                        if content.strip():
                            loaded_json = json.loads(content)
                            if isinstance(loaded_json, list):
                                all_feedback = loaded_json
                            else:
                                logger.warning(f"Feedback file {feedback_file} is not a list. Overwriting.")
                        # If file is empty or whitespace, all_feedback remains []
                    except json.JSONDecodeError:
                         logger.warning(f"Could not decode JSON from {feedback_file}. Overwriting.")
                    except Exception as read_err:
                        logger.error(f"Error reading feedback file {feedback_file}: {read_err}")
                        # Decide if this is fatal. For now, proceed to overwrite/create.

            # Add new feedback
            all_feedback.append(feedback_data)

            # Write back to file
            with open(feedback_file, 'w') as f:
                json.dump(all_feedback, f, indent=2)

        except Exception as file_error:
            logger.error(f"Error saving feedback to file: {str(file_error)}")

        # Flush metrics after recording feedback
        flush_metrics()

        return {"success": True, "message": "Feedback recorded successfully"}

    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        # Use .get() for session_id as it might be Optional in the request model
        increment_counter('errors', session_id=request.session_id, metric_name='feedback_error') # Specific error counter
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

# --- END: app.py /feedback Endpoint ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for AWS load balancer
    """
    return {
        "status": "healthy",
        "version": app.version,
        "environment": os.getenv("ENVIRONMENT", "production")
    }

# --- START: app.py /dashboard/metrics Endpoint ---

@app.get("/dashboard/metrics", status_code=200)
async def get_dashboard_metrics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    session_id_filter: Optional[str] = Query(None, alias="session_id", description="Filter by Session ID")
):
    """
    Get aggregated metrics by reading and processing the raw event file (metrics.json).
    Allows filtering by date and session ID.
    """
    logger.info(f"Fetching dashboard metrics. Filters: start={start_date}, end={end_date}, session={session_id_filter}")
    try:
        metrics_file_path = "metrics.json"
        raw_events: List[Dict[str, Any]] = []

        # --- Step 1: Read the Raw Event Data File ---
        if os.path.exists(metrics_file_path):
            try:
                with open(metrics_file_path, 'r') as f:
                    content = f.read()
                    if content.strip():
                        loaded_json = json.loads(content)
                        if isinstance(loaded_json, list):
                           raw_events = loaded_json
                        else:
                           logger.warning(f"Metrics file {metrics_file_path} does not contain a list. Treating as empty.")
                    else:
                         logger.info(f"Metrics file {metrics_file_path} is empty.")
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from {metrics_file_path}. Treating as empty.")
            except Exception as read_err:
                 logger.error(f"Error reading metrics file {metrics_file_path}: {read_err}")
                 # Continue with empty data for dashboard robustness

        # --- Step 2: Prepare Filters ---
        filter_start_dt: Optional[datetime] = None
        filter_end_dt: Optional[datetime] = None

        if start_date:
            try:
                filter_start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                logger.warning(f"Invalid start_date format: {start_date}. Expected YYYY-MM-DD.")
        if end_date:
            try:
                filter_end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)
            except ValueError:
                logger.warning(f"Invalid end_date format: {end_date}. Expected YYYY-MM-DD.")

        # --- Step 3: Filter Raw Events ---
        filtered_events: List[Dict[str, Any]] = []
        parse_errors = 0
        for event in raw_events:
            ts_str = event.get('timestamp')
            if not isinstance(ts_str, str):
                 # logger.debug(f"Skipping event with invalid timestamp: {event}") # Can be noisy
                 parse_errors += 1
                 continue

            try:
                # Handle potential timezone info ('Z' or offset) robustly
                if ts_str.endswith('Z'):
                    ts_str = ts_str[:-1] + '+00:00'
                event_dt = datetime.fromisoformat(ts_str)

            except ValueError:
                 # Add fallback parsing if needed, but fromisoformat is preferred
                 parse_errors += 1
                 logger.warning(f"Could not parse timestamp: {ts_str}. Skipping event.")
                 continue

            # Apply Date Filters (make timezone-aware if necessary, assuming UTC for now)
            if filter_start_dt and event_dt.replace(tzinfo=None) < filter_start_dt: # Naive comparison if filter dates are naive
                continue
            if filter_end_dt and event_dt.replace(tzinfo=None) > filter_end_dt: # Naive comparison
                continue

            # Apply Session ID Filter
            if session_id_filter and event.get('session_id') != session_id_filter:
                continue

            filtered_events.append(event)

        if parse_errors > 0:
            logger.warning(f"Skipped {parse_errors} events due to timestamp parsing issues during filtering.")
        logger.info(f"Found {len(filtered_events)} events matching filters.")


        # --- Step 4: Aggregate Filtered Events ---
        total_user_messages: int = 0
        session_starts: set = set()
        response_times: List[float] = []
        query_processing_times: List[float] = []
        chatbot_api_calls: int = 0
        chatbot_tokens_used: int = 0
        feedback_positive: int = 0
        feedback_negative: int = 0
        safety_scores: List[float] = []
        empathy_scores: List[float] = []
        memory_usages: List[float] = []

        # --- Step 4.5: Aggregate Metrics by Day for Chart ---
        daily_aggregation = defaultdict(lambda: {
            'count': 0, # Count of responses evaluated for safety/empathy on this day
            'safety_scores': [],
            'empathy_scores': [],
            'response_times': [],
        })

        for event in filtered_events:
            event_type = event.get("event_type")
            metric_name = event.get("metric_name")
            value = event.get("value")
            session_id = event.get("session_id")
            ts_str = event.get('timestamp')

            # Aggregate overall stats
            if event_type == "counter":
                if isinstance(value, int):
                    if metric_name == "user_messages":
                        total_user_messages += value
                    elif metric_name == "session_starts" and session_id:
                        session_starts.add(session_id)
            elif event_type == "timer":
                if isinstance(value, (int, float)):
                    if metric_name == "total_response_time":
                        response_times.append(value)
                    elif metric_name == "query_processing":
                         query_processing_times.append(value)
            elif event_type == "api_call":
                if metric_name == "chatbot_response":
                     if isinstance(value, int): chatbot_api_calls += value
                     tokens = event.get("tokens_used", 0)
                     if isinstance(tokens, int): chatbot_tokens_used += tokens
            elif event_type == "feedback":
                 rating = event.get("rating") # Check rating for positive/negative if available
                 if rating is not None and isinstance(rating, int):
                     if rating >= 3: feedback_positive += 1 # Assuming 1-5 scale
                     else: feedback_negative += 1
                 elif isinstance(value, int): # Fallback to positive/negative flag if rating not present
                     if value == 1: feedback_positive += 1
                     elif value == 0: feedback_negative += 1
            elif event_type == "score":
                 if isinstance(value, (int, float)):
                     if metric_name == "safety_score": safety_scores.append(value)
                     elif metric_name == "empathy_score": empathy_scores.append(value)
            elif event_type == "measurement":
                 if isinstance(value, (int, float)):
                     if metric_name == "memory_usage_mb":
                         memory_usages.append(value)

            # Aggregate daily stats
            if ts_str:
                try:
                    if ts_str.endswith('Z'):
                       ts_str = ts_str[:-1] + '+00:00'
                    event_dt = datetime.fromisoformat(ts_str)
                    day_str = event_dt.strftime('%Y-%m-%d') # Group by day

                    if event_type == "score":
                        if isinstance(value, (int, float)):
                            if metric_name == "safety_score":
                                daily_aggregation[day_str]['safety_scores'].append(value)
                                daily_aggregation[day_str]['count'] += 1 # Count responses with safety score
                            elif metric_name == "empathy_score":
                                daily_aggregation[day_str]['empathy_scores'].append(value)
                                # Note: Count isn't incremented here, assuming safety is the primary driver for daily eval count
                    elif event_type == "timer":
                        if metric_name == "total_response_time" and isinstance(value, (int, float)):
                            daily_aggregation[day_str]['response_times'].append(value)

                except ValueError:
                    # Already logged during filtering pass if it failed there
                    pass
                except Exception as daily_agg_err:
                    logger.warning(f"Skipping event during daily aggregation due to error: {daily_agg_err} - Event: {event}")
                    continue

        # --- Process daily data for the chart ---
        daily_chart_data = {
            'dates': [],
            'avg_safety': [], # Corresponds to 'daily_safety' in HTML attempt (0-100%)
            'avg_empathy': [], # Can be used for 'daily_scores' in HTML if desired (0-1 score)
            'avg_response_time_ms': [] # Convert to ms for display
        }
        sorted_dates = sorted(daily_aggregation.keys())

        for day in sorted_dates:
            day_data = daily_aggregation[day]
            daily_chart_data['dates'].append(day)

            avg_safety_day = sum(day_data['safety_scores']) / len(day_data['safety_scores']) if day_data['safety_scores'] else 0
            daily_chart_data['avg_safety'].append(round(avg_safety_day * 100, 1)) # Convert to percentage 0-100

            avg_empathy_day = sum(day_data['empathy_scores']) / len(day_data['empathy_scores']) if day_data['empathy_scores'] else 0
            daily_chart_data['avg_empathy'].append(round(avg_empathy_day, 3)) # Keep as 0-1 score

            avg_resp_time_day = sum(day_data['response_times']) / len(day_data['response_times']) if day_data['response_times'] else 0
            daily_chart_data['avg_response_time_ms'].append(round(avg_resp_time_day * 1000)) # Convert s to ms

        # --- Step 5: Calculate Final Aggregated Values ---
        total_conversations = len(session_starts)
        total_responses_evaluated = len(safety_scores) # Use safety score count as proxy for evaluated responses

        avg_messages_per_conversation = (total_user_messages / total_conversations) if total_conversations > 0 else 0

        total_feedback = feedback_positive + feedback_negative
        improvement_rate = (feedback_positive / total_feedback * 100) if total_feedback > 0 else 0 # Renamed from positive_percentage

        avg_response_time_s = sum(response_times) / len(response_times) if response_times else 0
        min_response_time_s = min(response_times) if response_times else 0
        max_response_time_s = max(response_times) if response_times else 0

        avg_token_usage = (chatbot_tokens_used / chatbot_api_calls) if chatbot_api_calls > 0 else 0

        # Quality Metrics
        avg_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 0
        avg_empathy = sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0
        # Calculate Safety Rate (e.g., % of responses with safety score >= 0.9)
        safety_threshold = 0.9 # Define your threshold for "safe"
        safe_count = sum(1 for score in safety_scores if score >= safety_threshold)
        safety_rate = (safe_count / total_responses_evaluated * 100) if total_responses_evaluated > 0 else 0

        # Placeholder for Avg Score (using avg of safety and empathy for now)
        avg_score_placeholder = ((avg_safety + avg_empathy) / 2 * 10) if (avg_safety > 0 or avg_empathy > 0) else 0 # Scale 0-10

        avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else 0
        min_memory = min(memory_usages) if memory_usages else 0
        max_memory = max(memory_usages) if memory_usages else 0

        # --- Step 6: Format Output for Dashboard ---
        dashboard_data = {
            "summary": {
                "total_evaluations": total_user_messages, # Use user messages as total evaluations
                "total_conversations": total_conversations,
                "avg_messages_per_conversation": round(avg_messages_per_conversation, 1),
                "improvement_rate": round(improvement_rate, 1), # % based on positive feedback
                "safe_count": safe_count, # Needed for subtext
                "total_responses_evaluated": total_responses_evaluated, # Needed for subtext
                "positive_feedback_count": feedback_positive, # Needed for subtext
            },
            "quality_metrics": {
                 "avg_score": round(avg_score_placeholder, 1), # Placeholder average score (0-10)
                 "safety_rate": round(safety_rate, 1), # % of safe responses (0-100)
                 "avg_safety_score": round(avg_safety, 3), # Raw average safety (0-1)
                 "avg_empathy_score": round(avg_empathy, 3), # Raw average empathy (0-1)
                 # --- Relevance, Accuracy, etc. keys are removed ---
            },
            "performance_metrics": { # Group performance related items
                "response_times_ms": { # Send in ms
                    "average": round(avg_response_time_s * 1000),
                    "min": round(min_response_time_s * 1000),
                    "max": round(max_response_time_s * 1000)
                },
                "token_usage": {
                    "average": round(avg_token_usage),
                    "min": 0, # Cannot determine min tokens per call easily
                    "max": chatbot_tokens_used # Total tokens in the filtered period
                },
                "memory_usage_mb": {
                    "average": round(avg_memory, 2),
                    "min": round(min_memory, 2),
                    "max": round(max_memory, 2)
                }
            },
            "daily_metrics_trend": daily_chart_data, # Use this for the line chart
            "feedback_summary": { # Separate feedback details
                 "positive": feedback_positive,
                 "negative": feedback_negative,
                 "total": total_feedback
            },
            # Placeholder for features not yet implemented
            "rag_performance": None, # For the 'Coming Soon' section
            "top_issues": [] # Return empty list as it's not calculated
        }
        logger.info("Successfully aggregated dashboard metrics.")
        return dashboard_data

    except Exception as e:
        logger.error(f"Error processing dashboard metrics: {str(e)}", exc_info=True)
        # Return a default structure that matches the expected keys to prevent JS errors
        return {
            "summary": {"total_evaluations": 0, "total_conversations": 0, "avg_messages_per_conversation": 0.0, "improvement_rate": 0.0, "safe_count": 0, "total_responses_evaluated": 0, "positive_feedback_count": 0},
            # --- Removed avg_relevance etc. from default quality_metrics ---
            "quality_metrics": {"avg_score": 0.0, "safety_rate": 0.0, "avg_safety_score": 0.0, "avg_empathy_score": 0.0},
            "performance_metrics": {
                "response_times_ms": {"average": 0, "min": 0, "max": 0},
                "token_usage": {"average": 0, "min": 0, "max": 0},
                "memory_usage_mb": {"average": 0.0, "min": 0.0, "max": 0.0}
            },
            "daily_metrics_trend": {"dates": [], "avg_safety": [], "avg_empathy": [], "avg_response_time_ms": []},
            "feedback_summary": {"positive": 0, "negative": 0, "total": 0},
            "rag_performance": None,
            "top_issues": []
        }

# --- END: app.py /dashboard/metrics Endpoint ---

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """
    Serve the admin dashboard interface
    """
    return templates.TemplateResponse(
        "admin-dashboard.html", 
        {
            "request": request
        }
    )

@app.post("/test-multi-aspect", response_model=ChatResponse)
async def test_multi_aspect(request: ChatRequest, 
                         processor: MultiAspectQueryProcessor = Depends(get_processor),
                         memory: MemoryManager = Depends(get_memory_manager)):
    """
    Test endpoint specifically for processing multi-aspect queries
    This ensures the message is processed as a multi-aspect query even if the classifier might not detect it as such
    """
    try:
        start_time = time.time()
        logger.info(f"Received multi-aspect test request: {request.message[:50]}...")
        
        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Store user message in memory
        memory.add_message(
            session_id=session_id,
            message=request.message,
            role="user",
            metadata={"is_multi_aspect_test": True, **(request.metadata or {})}
        )
        
        # Get conversation history
        conversation_history = memory.get_history(session_id)
        
        # Force multi-aspect classification
        forced_classification = {
            "primary_type": "knowledge",
            "is_multi_aspect": True,
            "confidence_scores": {
                "knowledge": 0.7,
                "emotional": 0.6,
                "policy": 0.6
            },
            "topics": ["reproductive_health", "abortion", "state_laws", "emotional_support"],
            "sensitive_content": ["abortion"],
            "contains_location": True,
            "detected_locations": ["state"],
            "query_complexity": "complex"
        }
        
        # Process with forced multi-aspect
        async def process_with_forced_multi_aspect():
            # First classify the message normally
            classification = await processor.unified_classifier.classify(
                request.message, 
                conversation_history
            )
            
            # Override is_multi_aspect flag
            classification["is_multi_aspect"] = True
            
            # Boost all confidence scores to encourage multiple aspect processing
            for aspect_type in classification.get("confidence_scores", {}):
                classification["confidence_scores"][aspect_type] = max(
                    classification["confidence_scores"].get(aspect_type, 0), 
                    0.6
                )
                
            # Force query complexity to complex
            classification["query_complexity"] = "complex"
            
            # Decompose into aspects
            aspects = await processor.aspect_decomposer.decompose(
                request.message, 
                classification, 
                conversation_history
            )
            
            # Process each aspect
            aspect_responses = []
            aspect_tasks = []
            
            for aspect in aspects:
                aspect_type = aspect.get("type", "knowledge")
                
                if aspect_type in processor.handlers:
                    # Create processing task for this aspect
                    handler = processor.handlers[aspect_type]
                    task = asyncio.create_task(
                        processor._process_aspect(
                            handler=handler,
                            aspect=aspect,
                            message=request.message,
                            conversation_history=conversation_history,
                            user_location=request.user_location,
                            aspect_type=aspect_type
                        )
                    )
                    aspect_tasks.append(task)
            
            # Wait for all aspect processing to complete
            if aspect_tasks:
                aspect_responses = await asyncio.gather(*aspect_tasks)
                # Filter out None responses
                aspect_responses = [r for r in aspect_responses if r is not None]
            
            # Compose the final response
            response = processor.response_composer.compose_response(
                message=request.message,
                aspect_responses=aspect_responses,
                classification=classification
            )
            
            return response
        
        # Process the query with forced multi-aspect
        response_data = await process_with_forced_multi_aspect()
        
        # Add message ID and session ID to response
        message_id = str(uuid.uuid4())
        response_data["message_id"] = message_id
        response_data["session_id"] = session_id
        
        if "processing_time" not in response_data:
            response_data["processing_time"] = time.time() - start_time
        
        # Add debug info
        response_data["metadata"] = {
            "is_multi_aspect_test": True,
            "aspects_count": len(response_data.get("aspect_responses", [])) if "aspect_responses" in response_data else 0,
            **(response_data.get("metadata", {}))
        }
        
        # Store bot response in memory
        memory.add_message(
            session_id=session_id,
            message=response_data["text"],
            role="assistant",
            metadata={
                "message_id": message_id,
                "citations": response_data.get("citations", []),
                "citation_objects": response_data.get("citation_objects", []),
                "is_multi_aspect_test": True
            }
        )
        
        # Add empty graphics array for compatibility with frontend
        if "graphics" not in response_data:
            response_data["graphics"] = []
            
        # Extract state information from aspect_responses if available
        if "aspect_responses" in response_data:
            for aspect_response in response_data["aspect_responses"]:
                if aspect_response.get("aspect_type") == "policy":
                    # Copy state_code if it exists in this aspect response
                    if "state_code" in aspect_response and "state_code" not in response_data:
                        response_data["state_code"] = aspect_response["state_code"]
                        logger.info(f"Added state_code to response: {aspect_response['state_code']}")
                    
                    # Copy state_codes if it exists in this aspect response
                    if "state_codes" in aspect_response and "state_codes" not in response_data:
                        response_data["state_codes"] = aspect_response["state_codes"]
                        logger.info(f"Added state_codes to response: {aspect_response['state_codes']}")
                    
                    break
            
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing multi-aspect test: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/test-knowledge")
async def test_knowledge(request: ChatRequest):
    """
    Test endpoint to directly access the knowledge handler
    """
    try:
        logger.info(f"Testing knowledge handler with query: {request.message[:100]}")
        
        # Access the knowledge handler via the processor
        knowledge_handler = query_processor.handlers.get("knowledge")
        
        if not knowledge_handler:
            return {"error": "Knowledge handler not found"}
        
        # Process the query through the knowledge handler
        response = await knowledge_handler.process_query(
            query=request.message,
            full_message=request.message
        )
        
        # Log information about citations
        logger.info(f"Knowledge response has {len(response.get('citations', []))} citations")
        logger.info(f"Knowledge citation objects: {json.dumps(response.get('citation_objects', []))}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error testing knowledge handler: {str(e)}", exc_info=True)
        return {"error": f"Error testing knowledge handler: {str(e)}"}

@app.post("/test-citations")
async def test_citations(request: ChatRequest):
    """
    Test endpoint to directly create a response with citations
    """
    try:
        # Create a test response with explicit citation objects
        response = {
            "text": "Yes, it is normal to feel both relief and sadness after an abortion. Feeling relieved not to be pregnant and yet sad at the same time can be a confusing combination, but is common and understandable. Some women will know 'in their head' that having an abortion was the right decision and do not regret it, but at the same time feel sad 'in their heart' about the end of the pregnancy. However through time and talking with others this can resolve.",
            "citations": ["Planned Parenthood", "Sexual Health Sheffield"],
            "citation_objects": [
                {
                    "source": "Sexual Health Sheffield",
                    "url": "https://www.sexualhealthsheffield.nhs.uk/wp-content/uploads/2019/06/Wellbeing-after-an-abortion.pdf",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood",
                    "url": "https://www.plannedparenthood.org/learn/abortion",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time()
        }
        
        # Add empty graphics array for compatibility with frontend
        if "graphics" not in response:
            response["graphics"] = []
            
        logger.info(f"Test citation response created with {len(response['citation_objects'])} citation objects")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test citations: {str(e)}"}

@app.post("/test-duplicate-citations")
async def test_duplicate_citations(request: ChatRequest):
    """
    Test endpoint that creates a response with citation URLs
    """
    try:
        # Create a test response with meaningful citation objects
        response = {
            "text": "This is a test response with multiple citations from different sources. Pregnancy occurs when sperm fertilizes an egg, which can happen during unprotected vaginal sex. A woman is most fertile during ovulation, which typically occurs around the middle of her menstrual cycle. After fertilization, the fertilized egg (zygote) travels to the uterus and implants in the uterine lining, beginning pregnancy.",
            "citations": ["Planned Parenthood", "Mayo Clinic", "WebMD"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood",
                    "title": "How Pregnancy Happens",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/how-pregnancy-happens",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Mayo Clinic",
                    "title": "Getting Pregnant",
                    "url": "https://www.mayoclinic.org/healthy-lifestyle/getting-pregnant/in-depth/how-to-get-pregnant/art-20047611",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "WebMD",
                    "title": "Understanding Early Pregnancy",
                    "url": "https://www.webmd.com/baby/understanding-conception",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        logger.info(f"Test citation response created with {len(response['citation_objects'])} citation objects")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test citations: {str(e)}"}

@app.post("/test-inline-citations")
async def test_inline_citations(request: ChatRequest):
    """
    Test endpoint that creates a response with inline numbered citations
    """
    try:
        # Create a test response with explicit citation objects and numbered references
        response = {
            "text": "Pregnancy occurs when sperm fertilizes an egg [1], which can happen during unprotected vaginal sex. A woman is most fertile during ovulation [2], which typically occurs around the middle of her menstrual cycle. After fertilization, the fertilized egg (zygote) travels to the uterus and implants in the uterine lining, beginning pregnancy. For more detailed information, you can refer to resources from Planned Parenthood [1], Mayo Clinic [2], and WebMD [3].",
            "citations": ["Planned Parenthood", "Mayo Clinic", "WebMD"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood",
                    "title": "How Pregnancy Happens",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/how-pregnancy-happens",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Mayo Clinic",
                    "title": "Getting Pregnant",
                    "url": "https://www.mayoclinic.org/healthy-lifestyle/getting-pregnant/in-depth/how-to-get-pregnant/art-20047611",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "WebMD",
                    "title": "Understanding Early Pregnancy",
                    "url": "https://www.webmd.com/baby/understanding-conception",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        logger.info(f"Test inline citation response created with {len(response['citation_objects'])} citation objects")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test inline citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test inline citations: {str(e)}"}

@app.post("/test-planned-parenthood-citations")
async def test_planned_parenthood_citations(request: ChatRequest):
    """
    Test endpoint specifically for Planned Parenthood citations that match the log example
    """
    try:
        # Create a test response that matches the logs in the user's example
        response = {
            "text": "Pregnancy occurs when sperm fertilizes an egg, which typically happens during unprotected vaginal sex. For more detailed information, you can refer to resources from Planned Parenthood [1][2][3].",
            "citations": ["Planned Parenthood", "Planned Parenthood", "Planned Parenthood"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood",
                    "title": "How Pregnancy Happens",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/how-pregnancy-happens",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood",
                    "title": "I Think I'm Pregnant - Now What?",
                    "url": "https://www.plannedparenthood.org/learn/teens/stds-birth-control-pregnancy/i-think-im-pregnant-now-what",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood",
                    "title": "How Pregnancy Happens",  # Duplicate title to match the example
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/how-pregnancy-happens",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        logger.info(f"Test Planned Parenthood citation response created with {len(response['citation_objects'])} citation objects")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test Planned Parenthood citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test Planned Parenthood citations: {str(e)}"}

@app.post("/test-improved-citations")
async def test_improved_citations(request: ChatRequest):
    """
    Test endpoint with well-formatted citation objects and text
    """
    try:
        # Create a test response with proper citation handling and multiple sources
        response = {
            "text": "Emergency contraception (EC) works by preventing or delaying ovulation. There are different types of EC available, including Plan B One-Step, ella, and copper IUDs. For maximum effectiveness, emergency contraception should be used as soon as possible after unprotected sex.",
            "citations": ["Planned Parenthood", "Mayo Clinic", "WebMD"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood",
                    "title": "Emergency Contraception Options",
                    "url": "https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception/which-kind-emergency-contraception-should-i-use",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Mayo Clinic",
                    "title": "Morning-After Pill",  
                    "url": "https://www.mayoclinic.org/tests-procedures/morning-after-pill/about/pac-20394730",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "WebMD",
                    "title": "Emergency Contraception",
                    "url": "https://www.webmd.com/sex/birth-control/emergency-contraception",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        # Log URLs for debugging
        logger.info(f"Test improved citations created with {len(response['citation_objects'])} citation objects")
        logger.info("These sources should appear in the UI:")
        for i, citation in enumerate(response["citation_objects"]):
            logger.info(f"  - {citation['source']}: {citation['url']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test improved citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test improved citations: {str(e)}"}

@app.post("/test-clean-citations")
async def test_clean_citations(request: ChatRequest):
    """
    Test endpoint with clean text and separate citations
    """
    try:
        # Create a test response with citations in a format that will be processed by the frontend
        raw_text = """Preventing sexually transmitted infections (STIs) involves a combination of strategies aimed at reducing risk during sexual activity. Here are key prevention methods:
1. **Abstinence**: The only 100% effective way to avoid STIs is to abstain from any sexual contact, including vaginal, anal, and oral sex, as well as skin-to-skin genital touching (Planned Parenthood, SOURCE).
2. **Use of Barriers**: If you choose to have sex, using condoms (external or internal) and dental dams can significantly lower your risk of STIs. These barriers help block the exchange of bodily fluids and reduce skin-to-skin contact that can transmit infections (SOURCE Mayo Clinic).
3. **Regular Testing**: Getting tested for STIs regularly is crucial, especially if you have multiple partners or engage in unprotected sex. Early detection allows for treatment, which helps maintain your health and prevents the spread of infections to others (SOURCE CDC)."""
        
        response = {
            "text": raw_text,  # This will be cleaned by the frontend
            "citations": ["Planned Parenthood", "Mayo Clinic", "CDC"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood",
                    "title": "STDs & Safer Sex",
                    "url": "https://www.plannedparenthood.org/learn/stds-hiv-safer-sex",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Mayo Clinic",
                    "title": "Sexually transmitted disease (STD) prevention",  
                    "url": "https://www.mayoclinic.org/diseases-conditions/sexually-transmitted-diseases-stds/in-depth/std-prevention/art-20044293",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "CDC",
                    "title": "How You Can Prevent Sexually Transmitted Diseases",
                    "url": "https://www.cdc.gov/std/prevention/default.htm",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        # Log the citation objects for debugging
        logger.info("Test clean numbered citations created")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test clean citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test clean citations: {str(e)}"}

@app.on_event("startup")
async def startup_event():
    """
    Initialize components on startup
    """
    logger.info("Starting up Abby Chatbot API")
    
    # Initialize metrics collection -- added 3.22
    increment_counter('server_start')

    # In the future, we might want to initialize more components here
    # For example, loading pre-trained models, connecting to databases, etc.

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up on shutdown
    """
    logger.info("Shutting down Abby Chatbot API")
    
    # Clean up inactive sessions
    session_count = memory_manager.cleanup_inactive_sessions()
    logger.info(f"Cleaned up {session_count} inactive sessions")

@app.post("/test-duplicated-sources")
async def test_duplicated_sources(request: ChatRequest):
    """
    Test endpoint for duplicated sources with different URLs
    """
    try:
        # Create a test response with multiple citations from the same source but different URLs
        raw_text = """The signs of pregnancy can vary from person to person, and while some may experience symptoms early on, others may not notice any at all. Here are some common early signs of pregnancy:

1. **Missed Period**: This is often the first sign that prompts individuals to consider the possibility of pregnancy. [1]
2. **Swollen or Tender Breasts**: Hormonal changes can cause breast tenderness and swelling. [1]
3. **Nausea and/or Vomiting**: Commonly referred to as "morning sickness," this can occur at any time of the day. [2]
4. **Fatigue**: Many people feel unusually tired during early pregnancy due to hormonal changes. [1]
5. **Bloating**: Some may experience bloating similar to what is felt during PMS. [3]
6. **Constipation**: Hormonal changes can slow down the digestive system. [2]
7. **Frequent Urination**: Increased urination can occur as the uterus expands and puts pressure on the bladder. [3]

It's important to note that these symptoms can also be caused by other factors, such as stress or hormonal fluctuations unrelated to pregnancy. Therefore, the only definitive way to confirm a pregnancy is by taking a pregnancy test, which can be done at home or at a healthcare provider's office. [1]

If you suspect you might be pregnant, consider taking a pregnancy test after your missed period for the most accurate results. If the test is positive, it's advisable to schedule an appointment with a healthcare provider to discuss your options and ensure your health. [2]"""
        
        # Create citation objects with specific URLs
        citations = [
            {
                "source": "Planned Parenthood",
                "title": "Pregnancy Symptoms",
                "url": "https://www.plannedparenthood.org/learn/pregnancy/pregnancy-symptoms",
                "accessed_date": datetime.now().strftime('%Y-%m-%d')
            },
            {
                "source": "Planned Parenthood",
                "title": "Morning Sickness & Nausea During Pregnancy",
                "url": "https://www.plannedparenthood.org/learn/pregnancy/morning-sickness",
                "accessed_date": datetime.now().strftime('%Y-%m-%d')
            },
            {
                "source": "Planned Parenthood",
                "title": "Pregnancy Tests & Other Services",
                "url": "https://www.plannedparenthood.org/learn/pregnancy/pregnancy-tests",
                "accessed_date": datetime.now().strftime('%Y-%m-%d')
            }
        ]
        
        # Log each citation and its URL for debugging
        for i, citation in enumerate(citations):
            logger.info(f"Citation {i+1}: {citation['title']} - {citation['url']}")
        
        response = {
            "text": raw_text,
            "citations": ["Planned Parenthood", "Planned Parenthood", "Planned Parenthood"],
            "citation_objects": citations,
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        # Log the citation objects for debugging
        logger.info("Test duplicated sources created")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test duplicated sources: {str(e)}", exc_info=True)
        return {"error": f"Error creating test duplicated sources: {str(e)}"}

@app.post("/test-citation-links")
async def test_citation_links(request: ChatRequest):
    """
    Test endpoint specifically for citation links in text
    """
    try:
        # Create a test response with clear numbered citations and distinct URLs
        raw_text = """Pregnancy occurs when a sperm cell fertilizes an egg, leading to the implantation of the fertilized egg in the lining of the uterus. Here's a step-by-step overview of how this process happens:

1. **Ovulation**: About halfway through a woman's menstrual cycle, a mature egg is released from the ovary in a process called ovulation. The egg then travels through the fallopian tube towards the uterus. [1]

2. **Fertilization**: If sperm are present in the vagina (usually from vaginal intercourse), they can swim through the cervix and into the uterus, eventually reaching the fallopian tubes. If a sperm cell meets the egg within about 12-24 hours after ovulation, fertilization occurs. It only takes one sperm to fertilize the egg. [2]

3. **Development of the Fertilized Egg**: After fertilization, the fertilized egg (now called a zygote) begins to divide and grow as it moves down the fallopian tube toward the uterus. This process takes about 3-4 days. [3]

4. **Implantation**: Once the fertilized egg reaches the uterus, it floats for a couple of days before implanting itself into the thick, spongy lining of the uterus. This implantation usually occurs about 6-10 days after fertilization and marks the official start of pregnancy. [2]

5. **Hormonal Changes**: After implantation, the developing embryo releases hormones that prevent the uterine lining from shedding, which is why menstruation does not occur during pregnancy. [1]

If the egg is not fertilized or if the fertilized egg does not implant successfully, the body will shed the uterine lining during menstruation."""
        
        response = {
            "text": raw_text,
            "citations": ["Planned Parenthood", "Planned Parenthood", "Planned Parenthood"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood",
                    "title": "How Pregnancy Happens",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/how-pregnancy-happens",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood",
                    "title": "What happens during fertilization?",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/fertility/what-happens-fertilization",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood",
                    "title": "Pregnancy Tests & Care",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/pregnancy-tests",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        # Log detailed information about each citation for debugging
        logger.info("=== TEST CITATION LINKS RESPONSE ===")
        logger.info(f"Text has {len(response['citation_objects'])} citation objects")
        
        for i, citation in enumerate(response["citation_objects"]):
            logger.info(f"Citation {i+1}:")
            logger.info(f"  - Source: {citation['source']}")
            logger.info(f"  - Title: {citation['title']}")
            logger.info(f"  - URL: {citation['url']}")
            
        return response
        
    except Exception as e:
        logger.error(f"Error creating test citation links: {str(e)}", exc_info=True)
        return {"error": f"Error creating test citation links: {str(e)}"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)