# This file contains database models for the application
# Currently implementing message and feedback models

import datetime
import uuid
import json
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

logger = logging.getLogger(__name__)

class User(UserMixin):
    """
    User model for authentication
    Implements UserMixin for Flask-Login functionality
    """
    USERS_FILE = 'users.json'
    
    def __init__(self, username, password_hash, is_admin=False, user_id=None):
        self.id = user_id or str(uuid.uuid4())
        self.username = username
        self.password_hash = password_hash
        self.is_admin = is_admin
        logger.debug(f"Created User instance: {self.username} (admin: {self.is_admin})")
    
    @staticmethod
    def get_user(user_id):
        """Get user by ID"""
        logger.debug(f"Attempting to get user by ID: {user_id}")
        users = User.get_all_users()
        for user in users:
            if user.id == user_id:
                logger.debug(f"Found user by ID: {user.username}")
                return user
        logger.warning(f"No user found with ID: {user_id}")
        return None
    
    @staticmethod
    def get_by_username(username):
        """Get user by username"""
        logger.debug(f"Attempting to get user by username: {username}")
        users = User.get_all_users()
        for user in users:
            if user.username == username:
                logger.debug(f"Found user by username: {username}")
                return user
        logger.warning(f"No user found with username: {username}")
        return None
    
    @staticmethod
    def get_all_users():
        """Get all users"""
        logger.debug("Attempting to get all users")
        if not os.path.exists(User.USERS_FILE):
            logger.info("Users file not found, creating default admin user")
            # Create default admin if no users exist
            admin = User('admin', generate_password_hash('admin'), True)
            User.save_users([admin])
            return [admin]
        
        try:
            with open(User.USERS_FILE, 'r') as f:
                users_data = json.load(f)
                users = []
                for user_data in users_data:
                    user = User(
                        user_data['username'],
                        user_data['password_hash'],
                        user_data['is_admin'],
                        user_data['id']
                    )
                    users.append(user)
                logger.debug(f"Loaded {len(users)} users from file")
                return users
        except Exception as e:
            logger.error(f"Error loading users: {str(e)}", exc_info=True)
            return []
    
    @staticmethod
    def save_users(users):
        """Save users to file"""
        logger.debug(f"Attempting to save {len(users)} users")
        users_data = []
        for user in users:
            users_data.append({
                'id': user.id,
                'username': user.username,
                'password_hash': user.password_hash,
                'is_admin': user.is_admin
            })
        
        try:
            with open(User.USERS_FILE, 'w') as f:
                json.dump(users_data, f, indent=4)
            logger.debug("Successfully saved users to file")
            return True
        except Exception as e:
            logger.error(f"Error saving users: {str(e)}", exc_info=True)
            return False
    
    def check_password(self, password):
        """Check password"""
        logger.debug(f"Checking password for user: {self.username}")
        result = check_password_hash(self.password_hash, password)
        logger.debug(f"Password check result: {result}")
        return result
    
    def to_dict(self):
        """Convert user to dictionary format"""
        return {
            'id': self.id,
            'username': self.username,
            'is_admin': self.is_admin
        }

class ChatMessage:
    """Represents a message in the chat conversation"""
    def __init__(self, content, sender, message_id=None, timestamp=None):
        self.content = content
        self.sender = sender  # 'user' or 'bot'
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.datetime.now()
        
    def to_dict(self):
        """Convert message to dictionary format"""
        return {
            "message_id": self.message_id,
            "content": self.content,
            "sender": self.sender,
            "timestamp": self.timestamp.isoformat()
        }

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
