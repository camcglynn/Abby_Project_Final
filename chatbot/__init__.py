# This file makes the chatbot directory a Python package

"""
Abby Chatbot - Multi-aspect query handling for reproductive health support
"""

from .multi_aspect_processor import MultiAspectQueryProcessor
from .memory_manager import MemoryManager
from .unified_classifier import UnifiedClassifier
from .aspect_decomposer import AspectDecomposer
from .knowledge_handler import KnowledgeHandler
from .emotional_support_handler import EmotionalSupportHandler
from .policy_handler import PolicyHandler
from .response_composer import ResponseComposer
from .preprocessor import Preprocessor

__version__ = "2.0.0"

__all__ = [
    'MultiAspectQueryProcessor',
    'MemoryManager',
    'UnifiedClassifier',
    'AspectDecomposer',
    'KnowledgeHandler',
    'EmotionalSupportHandler',
    'PolicyHandler',
    'ResponseComposer',
    'Preprocessor'
]
