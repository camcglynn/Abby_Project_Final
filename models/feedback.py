"""
Feedback model for storing user feedback on chatbot responses.
"""

import datetime

class UserFeedback:
    """Represents user feedback on a chatbot response"""
    def __init__(self, message_id, rating, comment=None, timestamp=None):
        self.message_id = message_id  # ID of the message being rated
        self.rating = rating  # Positive (1) or negative (-1) rating
        self.comment = comment  # Optional comment from user
        self.timestamp = timestamp or datetime.datetime.now()
        
    def to_dict(self):
        """Convert feedback to dictionary for storage/retrieval"""
        return {
            "message_id": self.message_id,
            "rating": self.rating,
            "comment": self.comment,
            "timestamp": self.timestamp.isoformat()
        } 