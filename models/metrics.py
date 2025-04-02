import logging
import json
from datetime import datetime
import os

logger = logging.getLogger(__name__)

def get_metrics(start_date=None, end_date=None, session_id=None):
    """
    Get metrics data with optional filtering
    
    Args:
        start_date (datetime, optional): Start date filter
        end_date (datetime, optional): End date filter
        session_id (str, optional): Session ID filter
        
    Returns:
        dict: Dictionary of metrics data
    """
    try:
        # Sample metrics data - in a real implementation, this would come from a database
        metrics = {
            "total_conversations": 120,
            "total_messages": 542,
            "average_messages_per_conversation": 4.5,
            "positive_feedback_rate": 87.5,
            "average_response_time": 0.85,
            "most_common_topics": [
                {"topic": "Abortion Access", "count": 80},
                {"topic": "Contraception", "count": 65},
                {"topic": "Pregnancy", "count": 32},
                {"topic": "STI Information", "count": 24},
                {"topic": "General Questions", "count": 18}
            ],
            "bot_usage_by_state": [
                {"state": "NY", "count": 52},
                {"state": "CA", "count": 47},
                {"state": "TX", "count": 35},
                {"state": "FL", "count": 31},
                {"state": "IL", "count": 22}
            ],
            "message_history": []
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return {}

def format_metrics_for_display(metrics):
    """
    Format metrics data for display in templates
    
    Args:
        metrics (dict): Raw metrics data
        
    Returns:
        dict: Formatted metrics data
    """
    try:
        return {
            "overview": {
                "total_conversations": metrics.get("total_conversations", 0),
                "total_messages": metrics.get("total_messages", 0),
                "avg_messages_per_conversation": metrics.get("average_messages_per_conversation", 0),
                "positive_feedback_rate": f"{metrics.get('positive_feedback_rate', 0)}%",
                "avg_response_time": f"{metrics.get('average_response_time', 0):.2f}s"
            },
            "topics": metrics.get("most_common_topics", []),
            "states": metrics.get("bot_usage_by_state", []),
            "history": metrics.get("message_history", [])
        }
    except Exception as e:
        logger.error(f"Error formatting metrics: {str(e)}")
        return {} 