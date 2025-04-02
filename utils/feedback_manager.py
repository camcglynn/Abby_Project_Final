"""
Feedback Manager for the reproductive health chatbot

This module handles storing and retrieving user feedback
for chatbot responses to help improve the system.
"""

import json
import os
import logging
from datetime import datetime
from models import UserFeedback

logger = logging.getLogger(__name__)

class FeedbackManager:
    """Manages storing and retrieving user feedback for chatbot responses"""
    
    def __init__(self, feedback_file="user_feedback.json"):
        """Initialize the feedback manager"""
        self.feedback_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), feedback_file)
        self.feedback_data = self._load_feedback_data()
        
    def _load_feedback_data(self):
        """Load feedback data from storage file"""
        if not os.path.exists(self.feedback_file):
            logger.info(f"Feedback file {self.feedback_file} not found, creating new file")
            return []
            
        try:
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} feedback entries")
                return data
        except Exception as e:
            logger.error(f"Error loading feedback data: {str(e)}")
            return []
    
    def _save_feedback_data(self):
        """Save feedback data to storage file"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
            logger.info(f"Saved {len(self.feedback_data)} feedback entries")
            return True
        except Exception as e:
            logger.error(f"Error saving feedback data: {str(e)}")
            return False
    
    def add_feedback(self, message_id, rating, comment=None):
        """
        Add a new feedback entry
        
        Args:
            message_id (str): ID of the message being rated
            rating (int): 1 for thumbs up, -1 for thumbs down
            comment (str, optional): Optional comment from user
            
        Returns:
            bool: True if feedback was added successfully
        """
        try:
            # Double-check for PII in comment if provided
            if comment:
                from utils.text_processing import PIIDetector
                pii_detector = PIIDetector()
                
                if pii_detector.has_pii(comment):
                    logger.warning("PII detected in feedback comment during storage, redacting")
                    comment, _ = pii_detector.redact_pii(comment)
            
            # Create new feedback object
            feedback = UserFeedback(
                message_id=message_id,
                rating=rating,
                comment=comment
            )
            
            # Add to internal data structure
            self.feedback_data.append(feedback.to_dict())
            
            # Save to file
            success = self._save_feedback_data()
            
            return success
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            return False
    
    def get_feedback_for_message(self, message_id):
        """
        Get all feedback for a specific message
        
        Args:
            message_id (str): ID of the message
            
        Returns:
            list: List of feedback entries for the message
        """
        return [f for f in self.feedback_data if f.get('message_id') == message_id]
    
    def get_all_feedback(self):
        """
        Get all feedback entries
        
        Returns:
            list: List of all feedback entries
        """
        return self.feedback_data
    
    def get_feedback_stats(self):
        """
        Get statistics about feedback
        
        Returns:
            dict: Dictionary with feedback statistics
        """
        total = len(self.feedback_data)
        positive = sum(1 for f in self.feedback_data if f.get('rating', 0) > 0)
        negative = sum(1 for f in self.feedback_data if f.get('rating', 0) < 0)
        
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "positive_percentage": round((positive / total) * 100, 2) if total > 0 else 0,
            "negative_percentage": round((negative / total) * 100, 2) if total > 0 else 0
        }