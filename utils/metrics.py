print("--- LOADING utils/metrics.py ---") # <--- ADD THIS AT THE VERY TOP

# utils/metrics.py
import os
import time
import json
import logging
import threading
from datetime import datetime
from collections import defaultdict # Keep for potential future use, but not core storage
from typing import Dict, List, Any, Optional, Union

# Import boto3 conditionally (optional for raw logging)
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# Import psutil conditionally
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetricsTracker:
    """
    Tracks raw metric events about the chatbot's usage and performance.
    Events are periodically flushed to a JSON file.
    """
    def __init__(self,
                 app_name: str = "ReproductiveHealthChatbot", # Still useful for potential CW
                 metrics_file: str = "metrics.json",
                 auto_flush_interval: Optional[int] = 300, # Can be None to disable auto-flush
                 enable_cloudwatch: bool = False): # Defaulting CW to False for raw logs
        """
        Initialize the metrics tracker for raw event logging.

        Args:
            app_name (str): Application name (e.g., for CloudWatch namespace).
            metrics_file (str): Path to metrics file for local storage (list of events).
            auto_flush_interval (Optional[int]): Seconds between auto-flush to storage.
                                                  Set to None or 0 to disable.
            enable_cloudwatch (bool): Whether to attempt sending raw events to CloudWatch
                                      (Complex, potentially high volume, default False).
        """
        self.app_name = app_name
        self.metrics_file = metrics_file
        self.auto_flush_interval = auto_flush_interval
        self.enable_cloudwatch = enable_cloudwatch and BOTO3_AVAILABLE # Keep check

        self.cloudwatch = None
        if self.enable_cloudwatch:
            region = os.environ.get('AWS_REGION', 'us-east-1')
            try:
                self.cloudwatch = boto3.client('cloudwatch', region_name=region)
                logger.info(f"CloudWatch metrics enabled (region: {region}) - Note: Sending raw events.")
            except Exception as e:
                logger.warning(f"Failed to initialize CloudWatch client: {str(e)}")
                self.enable_cloudwatch = False

        # Store pending events in memory before flushing
        self._metrics_lock = threading.Lock()
        self._pending_events: List[Dict[str, Any]] = []

        # Set up auto-flushing if interval is valid
        if auto_flush_interval and auto_flush_interval > 0:
            self._setup_auto_flush()
        elif auto_flush_interval is not None:
             logger.info("Auto-flush disabled (interval <= 0). Manual flush needed.")


        if not PSUTIL_AVAILABLE:
             logger.warning("psutil not installed. Memory usage tracking disabled.")

        logger.info(f"Metrics tracker initialized for raw event logging: file='{self.metrics_file}', auto_flush={self.auto_flush_interval}s, cloudwatch={self.enable_cloudwatch}")

    def _add_event(self, event_data: Dict[str, Any]):
        """Internal method to add an event dictionary to the pending list."""
        # Ensure timestamp exists
        if 'timestamp' not in event_data:
            event_data['timestamp'] = datetime.now().isoformat()

        with self._metrics_lock:
            self._pending_events.append(event_data)

    def _setup_auto_flush(self):
        """Set up a timer to periodically flush metrics."""
        if not self.auto_flush_interval or self.auto_flush_interval <= 0:
             return

        def auto_flush_job():
            logger.debug("Auto-flush triggered.")
            self.flush_metrics()
            # Schedule the next flush using a new Timer thread
            if self.auto_flush_interval and self.auto_flush_interval > 0:
                timer = threading.Timer(self.auto_flush_interval, auto_flush_job)
                timer.daemon = True # Allow program to exit even if timer is pending
                timer.start()

        # Start the first timer
        logger.info(f"Auto-flush enabled (interval: {self.auto_flush_interval}s)")
        initial_timer = threading.Timer(self.auto_flush_interval, auto_flush_job)
        initial_timer.daemon = True
        initial_timer.start()

    # --- Raw Event Recording Methods ---

    def increment_counter(self, metric_name: str, value: int = 1, session_id: Optional[str] = None):
        """Record a counter event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "counter",
            "metric_name": metric_name,
            "value": value,
            "session_id": session_id
        }
        self._add_event(event)

    def record_time(self, metric_name: str, elapsed_time: float, session_id: Optional[str] = None, category: Optional[str] = None):
        """Record a timing event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "timer",
            "metric_name": metric_name,
            "value": elapsed_time, # Store the raw time
            "session_id": session_id,
            "category": category # Optional category
        }
        self._add_event(event)

    def record_api_call(self, api_name: str, tokens_used: int = 0, session_id: Optional[str] = None):
         """Record an API call event with token usage."""
         event = {
             "timestamp": datetime.now().isoformat(),
             "event_type": "api_call",
             "metric_name": api_name,
             "value": 1, # Represents one call
             "tokens_used": tokens_used, # Specific metadata
             "session_id": session_id
         }
         self._add_event(event)

    def record_feedback(self, positive: bool = True, message_id: Optional[str] = None, session_id: Optional[str] = None, rating: Optional[int] = None, comment: Optional[str] = None):
        """Record a feedback event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "feedback",
            "metric_name": "user_feedback",
            "value": 1 if positive else 0, # 1 for positive, 0 for negative
            "message_id": message_id, # Specific metadata
            "rating": rating,
            "comment": comment,
            "session_id": session_id
        }
        self._add_event(event)

    def record_safety_score(self, score: float, session_id: Optional[str] = None, message_id: Optional[str] = None):
        """Record a safety score event."""
        if not isinstance(score, (int, float)): return
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "score",
            "metric_name": "safety_score",
            "value": score,
            "session_id": session_id,
            "message_id": message_id
        }
        self._add_event(event)

    def record_empathy_score(self, score: float, session_id: Optional[str] = None, message_id: Optional[str] = None):
        """Record an empathy score event."""
        if not isinstance(score, (int, float)): return
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "score",
            "metric_name": "empathy_score",
            "value": score,
            "session_id": session_id,
            "message_id": message_id
        }
        self._add_event(event)

    def record_memory_usage(self, session_id: Optional[str] = None):
        """Record the current process's memory usage (RSS) in MB."""
        if not PSUTIL_AVAILABLE: return
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "measurement",
                "metric_name": "memory_usage_mb",
                "value": memory_mb,
                "session_id": session_id
            }
            self._add_event(event)
        except Exception as e:
            logger.warning(f"Could not record memory usage: {e}")

    def record_measurement(self, metric_name: str, value: Union[int, float], session_id: Optional[str] = None, message_id: Optional[str] = None):
        """Record a generic measurement event."""
        if not isinstance(value, (int, float)):
            logger.warning(f"Invalid measurement value type for {metric_name}: {type(value)}. Expected float or int.")
            return
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "measurement",
            "metric_name": metric_name,
            "value": value,
            "session_id": session_id,
            "message_id": message_id # Optional context
        }
        self._add_event(event)
    
    def flush_metrics(self):
        """
        Flush pending raw metric events to the storage file.
        Clears the pending events list after attempting to flush.
        """
        events_to_flush = []
        with self._metrics_lock:
            if not self._pending_events:
                logger.debug("No pending metrics to flush.")
                return # Nothing to do

            # Make a copy and clear the original list safely
            events_to_flush = list(self._pending_events) # Shallow copy is fine
            self._pending_events = []

        logger.info(f"Attempting to flush {len(events_to_flush)} raw metric events.")

        # --- Save to local file ---
        try:
            existing_events = []
            if os.path.exists(self.metrics_file):
                try:
                    with open(self.metrics_file, 'r') as f:
                        content = f.read()
                        # Handle empty file or file with just whitespace
                        if content.strip():
                            existing_events = json.loads(content)
                            if not isinstance(existing_events, list):
                                logger.warning(f"Metrics file {self.metrics_file} is not a list. Overwriting.")
                                existing_events = []
                        else:
                            existing_events = [] # Treat empty file as empty list
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in metrics file {self.metrics_file}. Overwriting.")
                    existing_events = []
                except Exception as read_err:
                    logger.error(f"Error reading metrics file {self.metrics_file}, data might be lost: {read_err}")
                    # Decide if you want to proceed and overwrite or stop
                    # For robustness, maybe try backing up the bad file?
                    # For now, we'll proceed and potentially overwrite.
                    existing_events = []

            # Append new events
            existing_events.extend(events_to_flush)

            # Write the entire updated list back
            with open(self.metrics_file, 'w') as f:
                json.dump(existing_events, f, indent=2) # indent=2 for readability, remove for smaller size

            logger.debug(f"Successfully flushed {len(events_to_flush)} events to {self.metrics_file}")

        except Exception as e:
            logger.error(f"Error saving metrics to file {self.metrics_file}: {str(e)}")
            # Consider adding unflushed events back to pending list? Or log them?
            # with self._metrics_lock:
            #     self._pending_events = events_to_flush + self._pending_events # Put them at the front


        # --- Send to CloudWatch (Optional, potentially high volume) ---
        if self.enable_cloudwatch and self.cloudwatch:
            # Sending raw events individually can be inefficient and costly.
            # Consider aggregating before sending or using CloudWatch Logs instead.
            # This simple example sends each event - USE WITH CAUTION.
            logger.warning("CloudWatch is enabled for raw events - this may generate high volume.")
            cw_metrics_data = []
            for event in events_to_flush:
                 try:
                     # Basic mapping - Needs refinement based on event_type
                     metric_name_cw = f"{event['event_type'].capitalize()}_{event['metric_name']}"
                     value_cw = event.get('value', 1) # Default value if missing?

                     if isinstance(value_cw, (int, float)):
                        cw_event = {
                            'MetricName': metric_name_cw,
                            'Value': value_cw,
                            # Add dimensions like session_id if needed:
                            # 'Dimensions': [{'Name': 'SessionId', 'Value': event.get('session_id','N/A')}],
                            # 'Timestamp': datetime.fromisoformat(event['timestamp']) # Optional: use event time
                        }
                        # Determine unit based on type/name (simplistic)
                        if event['event_type'] == 'timer':
                            cw_event['Unit'] = 'Seconds'
                        elif event['event_type'] == 'measurement' and 'mb' in event['metric_name'].lower():
                            cw_event['Unit'] = 'Megabytes'
                        elif event['event_type'] == 'counter' or event['event_type'] == 'api_call':
                             cw_event['Unit'] = 'Count'
                        # Add more unit logic as needed

                        cw_metrics_data.append(cw_event)
                 except Exception as cw_format_err:
                      logger.warning(f"Could not format event for CloudWatch: {cw_format_err} - Event: {event}")


            # Send in batches
            batch_size = 20
            for i in range(0, len(cw_metrics_data), batch_size):
                batch = cw_metrics_data[i:i+batch_size]
                try:
                    self.cloudwatch.put_metric_data(
                        Namespace=self.app_name,
                        MetricData=batch
                    )
                    logger.debug(f"Sent batch of {len(batch)} raw events to CloudWatch.")
                except Exception as e:
                    logger.error(f"Error sending metrics batch to CloudWatch: {str(e)}")

# --- Global instance and updated helper functions ---
metrics = MetricsTracker() # Configure parameters as needed

def increment_counter(metric_name: str, value: int = 1, session_id: Optional[str] = None):
    metrics.increment_counter(metric_name, value, session_id)

def record_time(metric_name: str, elapsed_time: float, session_id: Optional[str] = None, category: Optional[str] = None):
    metrics.record_time(metric_name, elapsed_time, session_id, category)

def record_api_call(api_name: str, tokens_used: int = 0, session_id: Optional[str] = None):
     metrics.record_api_call(api_name, tokens_used, session_id)

def record_feedback(positive: bool = True, message_id: Optional[str] = None, session_id: Optional[str] = None, rating: Optional[int] = None, comment: Optional[str] = None):
    metrics.record_feedback(positive, message_id, session_id, rating, comment)

# ----> MAKE SURE THIS FUNCTION IS PRESENT <----
def record_safety_score(score: float, session_id: Optional[str] = None, message_id: Optional[str] = None):
    metrics.record_safety_score(score, session_id, message_id)
# ----> AND THIS ONE <----
def record_empathy_score(score: float, session_id: Optional[str] = None, message_id: Optional[str] = None):
    metrics.record_empathy_score(score, session_id, message_id)
# ----> AND THIS ONE <----
def record_memory_usage(session_id: Optional[str] = None):
    metrics.record_memory_usage(session_id)
# ----> AND THIS ONE <----
def record_measurement(metric_name: str, value: Union[int, float], session_id: Optional[str] = None, message_id: Optional[str] = None):
    """Helper function to record a generic measurement event."""
    metrics.record_measurement(metric_name, value, session_id, message_id)

def flush_metrics():
    """Manually trigger a flush of pending metrics."""
    metrics.flush_metrics()