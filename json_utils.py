"""
JSON serialization utilities for handling NumPy data types and other non-serializable objects
"""

import json
import numpy as np
from datetime import datetime
from typing import Any


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy data types and other common non-serializable objects.
    """
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        return super().default(obj)


def safe_json_serialize(data: Any) -> Any:
    """
    Safely serialize data by converting NumPy types to native Python types.
    
    Args:
        data: Data structure to serialize
        
    Returns:
        JSON-serializable version of the data
    """
    if isinstance(data, dict):
        return {key: safe_json_serialize(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [safe_json_serialize(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(safe_json_serialize(item) for item in data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, datetime):
        return data.isoformat()
    elif hasattr(data, 'item'):  # Handle numpy scalars
        return data.item()
    else:
        return data


def clean_cluster_data(clusters: list) -> list:
    """
    Clean cluster data specifically to ensure JSON serialization.
    
    Args:
        clusters: List of cluster dictionaries
        
    Returns:
        Cleaned cluster data
    """
    cleaned_clusters = []
    
    for cluster in clusters:
        cleaned_cluster = {}
        
        for key, value in cluster.items():
            if key == 'results':
                # Clean each result in the cluster
                cleaned_results = []
                for result in value:
                    cleaned_result = safe_json_serialize(result)
                    cleaned_results.append(cleaned_result)
                cleaned_cluster[key] = cleaned_results
            else:
                # Clean other cluster properties
                cleaned_cluster[key] = safe_json_serialize(value)
        
        cleaned_clusters.append(cleaned_cluster)
    
    return cleaned_clusters


def clean_search_results(results: list) -> list:
    """
    Clean search results to ensure JSON serialization.
    
    Args:
        results: List of search result dictionaries
        
    Returns:
        Cleaned search results
    """
    return [safe_json_serialize(result) for result in results]


def clean_metrics_data(metrics: dict) -> dict:
    """
    Clean metrics data to ensure JSON serialization.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Cleaned metrics data
    """
    return safe_json_serialize(metrics)
