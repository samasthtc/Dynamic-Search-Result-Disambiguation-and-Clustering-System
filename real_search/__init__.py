"""
Real Search Package - Fixed Version
Complete real dataset-based search disambiguation system
"""

from .system import RealSearchSystem
from .datasets import DatasetManager
from .clustering import ClusteringEngine
from .feedback import FeedbackProcessor
from .json_utils import NumpyEncoder, clean_for_json

__version__ = "2.0.0"
__author__ = "Search Disambiguation Research Team"

# Main exports
__all__ = [
    "RealSearchSystem",
    "DatasetManager", 
    "ClusteringEngine",
    "FeedbackProcessor",
    "NumpyEncoder",
    "clean_for_json",
]