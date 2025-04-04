import logging
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from collections import deque
import os

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages conversation history and context for the chatbot.
    
    This class:
    1. Stores and retrieves conversation history
    2. Manages memory size to prevent context overload
    3. Extracts relevant context for queries
    4. Persists conversations for future reference
    """
    
    def __init__(self, 
                max_history_size: int = 20,
                persistence_path: Optional[str] = None):
        """
        Initialize the memory manager
        
        Args:
            max_history_size (int): Maximum number of messages to store in memory
            persistence_path (Optional[str]): Path to store conversation data, if None uses default
        """
        logger.info(f"Initializing MemoryManager with max_history_size={max_history_size}")
        
        self.max_history_size = max_history_size
        self.persistence_path = persistence_path or os.path.join(os.getcwd(), "data", "conversations")
        
        # Ensure persistence directory exists
        if not os.path.exists(self.persistence_path):
            os.makedirs(self.persistence_path, exist_ok=True)
            logger.info(f"Created persistence directory: {self.persistence_path}")
        
        # Active conversation sessions
        self.active_sessions = {}
    
    def add_message(self, 
                   session_id: str, 
                   message: str, 
                   role: str = "user", 
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a message to the conversation history
        
        Args:
            session_id (str): Unique identifier for the conversation
            message (str): The message text
            role (str): The role of the sender (user or assistant)
            metadata (Optional[Dict[str, Any]]): Additional message metadata
            
        Returns:
            Dict[str, Any]: The message object as stored
        """
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session: {session_id}")
        
        # Initialize session if it doesn't exist
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "messages": deque(maxlen=self.max_history_size),
                "created_at": time.time(),
                "last_active": time.time()
            }
        
        # Prepare message object
        message_obj = {
            "message_id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": message,
            "timestamp": time.time()
        }
        
        # Add metadata if provided
        if metadata:
            message_obj.update(metadata)
        
        # Add to session
        self.active_sessions[session_id]["messages"].append(message_obj)
        self.active_sessions[session_id]["last_active"] = time.time()
        
        # Persist the updated conversation
        self._persist_session(session_id)
        
        return message_obj
    
    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session
        
        Args:
            session_id (str): Unique identifier for the conversation
            limit (Optional[int]): Maximum number of messages to return, None for all
            
        Returns:
            List[Dict[str, Any]]: List of message objects
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Attempted to access nonexistent session: {session_id}")
            # Try to load from persistent storage
            if self._load_session(session_id):
                logger.info(f"Loaded session from persistent storage: {session_id}")
            else:
                logger.info(f"Creating new empty session: {session_id}")
                self.active_sessions[session_id] = {
                    "messages": deque(maxlen=self.max_history_size),
                    "created_at": time.time(),
                    "last_active": time.time()
                }
        
        # Update last active time
        self.active_sessions[session_id]["last_active"] = time.time()
        
        # Get messages
        messages = list(self.active_sessions[session_id]["messages"])
        
        # Apply limit if specified
        if limit and limit > 0:
            messages = messages[-limit:]
        
        return messages
    
    def get_relevant_context(self, 
                           session_id: str, 
                           query: str, 
                           max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        Get context most relevant to the current query
        
        Args:
            session_id (str): Unique identifier for the conversation
            query (str): The current query to find relevant context for
            max_messages (int): Maximum number of messages to include
            
        Returns:
            List[Dict[str, Any]]: List of relevant message objects
        """
        # For now, we're simply returning the most recent messages
        # In a more advanced implementation, this would use semantic search
        # to find the most relevant messages from the history
        
        # Get full history
        history = self.get_history(session_id)
        
        # Return most recent messages up to max_messages
        return history[-max_messages:] if history else []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a conversation session from memory
        
        Args:
            session_id (str): Unique identifier for the conversation
            
        Returns:
            bool: True if successful, False otherwise
        """
        if session_id in self.active_sessions:
            # Create an empty backup before clearing
            self._persist_session(session_id, backup=True)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"Cleared session: {session_id}")
            return True
        else:
            logger.warning(f"Attempted to clear nonexistent session: {session_id}")
            return False
    
    def clear_all_sessions(self) -> int:
        """
        Clear all active conversation sessions
        
        Returns:
            int: Number of sessions cleared
        """
        count = len(self.active_sessions)
        
        # Backup all sessions before clearing
        for session_id in list(self.active_sessions.keys()):
            self._persist_session(session_id, backup=True)
        
        # Clear all sessions
        self.active_sessions = {}
        
        logger.info(f"Cleared all {count} active sessions")
        return count
    
    def _persist_session(self, session_id: str, backup: bool = False) -> bool:
        """
        Persist a session to storage
        
        Args:
            session_id (str): Unique identifier for the conversation
            backup (bool): Whether this is a backup before deletion
            
        Returns:
            bool: True if successful, False otherwise
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Attempted to persist nonexistent session: {session_id}")
            return False
        
        try:
            # Create filename
            timestamp = int(time.time())
            suffix = f"_backup_{timestamp}" if backup else ""
            filename = f"{session_id}{suffix}.json"
            filepath = os.path.join(self.persistence_path, filename)
            
            # Prepare data for serialization
            session_data = {
                "session_id": session_id,
                "created_at": self.active_sessions[session_id]["created_at"],
                "last_active": self.active_sessions[session_id]["last_active"],
                "messages": list(self.active_sessions[session_id]["messages"])
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.debug(f"Persisted session to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error persisting session {session_id}: {str(e)}", exc_info=True)
            return False
    
    def _load_session(self, session_id: str) -> bool:
        """
        Load a session from persistent storage
        
        Args:
            session_id (str): Unique identifier for the conversation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Look for session file
            filepath = os.path.join(self.persistence_path, f"{session_id}.json")
            
            if not os.path.exists(filepath):
                logger.debug(f"No persistent storage found for session: {session_id}")
                return False
            
            # Read data
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            # Create session in memory
            self.active_sessions[session_id] = {
                "messages": deque(session_data["messages"], maxlen=self.max_history_size),
                "created_at": session_data.get("created_at", time.time()),
                "last_active": session_data.get("last_active", time.time())
            }
            
            logger.info(f"Loaded session {session_id} with {len(session_data['messages'])} messages")
            return True
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}", exc_info=True)
            return False
    
    def cleanup_inactive_sessions(self, max_age_hours: int = 24) -> int:
        """
        Remove inactive sessions from memory
        
        Args:
            max_age_hours (int): Maximum age in hours before a session is considered inactive
            
        Returns:
            int: Number of sessions cleaned up
        """
        max_age_seconds = max_age_hours * 3600
        current_time = time.time()
        
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            # Check if session is older than max_age
            if current_time - session["last_active"] > max_age_seconds:
                sessions_to_remove.append(session_id)
        
        # Remove inactive sessions
        for session_id in sessions_to_remove:
            # Persist before removing
            self._persist_session(session_id)
            del self.active_sessions[session_id]
        
        count = len(sessions_to_remove)
        if count > 0:
            logger.info(f"Cleaned up {count} inactive sessions")
        
        return count 