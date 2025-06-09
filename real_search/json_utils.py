"""
JSON Utilities
Handles JSON serialization of numpy arrays and complex data structures
"""

import json
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Union


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and other non-serializable types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)

        return super(NumpyEncoder, self).default(obj)


def clean_for_json(data: Any) -> Any:
    """
    Recursively clean data structure for JSON serialization

    Args:
        data: Data to clean

    Returns:
        JSON-serializable data
    """
    if isinstance(data, dict):
        return {key: clean_for_json(value) for key, value in data.items()}

    elif isinstance(data, list):
        return [clean_for_json(item) for item in data]

    elif isinstance(data, tuple):
        return [clean_for_json(item) for item in data]

    elif isinstance(data, set):
        return [clean_for_json(item) for item in data]

    elif isinstance(data, np.ndarray):
        return data.tolist()

    elif isinstance(data, (np.floating, np.complexfloating)):
        return float(data)

    elif isinstance(data, (np.integer, np.signedinteger, np.unsignedinteger)):
        return int(data)

    elif isinstance(data, np.bool_):
        return bool(data)

    elif isinstance(data, datetime):
        return data.isoformat()

    elif hasattr(data, "__dict__"):
        # Handle custom objects by converting to dict
        return clean_for_json(data.__dict__)

    else:
        # Return as-is for basic types (str, int, float, bool, None)
        return data


def safe_json_dumps(data: Any, **kwargs) -> str:
    """
    Safely serialize data to JSON string

    Args:
        data: Data to serialize
        **kwargs: Additional arguments for json.dumps

    Returns:
        JSON string
    """
    try:
        cleaned_data = clean_for_json(data)
        return json.dumps(cleaned_data, cls=NumpyEncoder, **kwargs)
    except Exception as e:
        # Fallback for problematic data
        return json.dumps(
            {
                "error": "Serialization failed",
                "error_type": str(type(e).__name__),
                "data_type": str(type(data).__name__),
            }
        )


def safe_json_loads(json_str: str) -> Any:
    """
    Safely deserialize JSON string

    Args:
        json_str: JSON string to deserialize

    Returns:
        Deserialized data or error dict
    """
    try:
        return json.loads(json_str)
    except Exception as e:
        return {"error": "Deserialization failed", "error_type": str(type(e).__name__)}


def clean_search_results(results: List[Dict]) -> List[Dict]:
    """
    Clean search results for JSON serialization

    Args:
        results: List of search result dictionaries

    Returns:
        Cleaned search results
    """
    cleaned_results = []

    for result in results:
        try:
            cleaned_result = clean_for_json(result)

            # Ensure required fields exist with defaults
            cleaned_result.setdefault("id", f"result_{len(cleaned_results)}")
            cleaned_result.setdefault("title", "Untitled")
            cleaned_result.setdefault("snippet", "")
            cleaned_result.setdefault("url", "")
            cleaned_result.setdefault("domain", "unknown.com")
            cleaned_result.setdefault("category", "general")
            cleaned_result.setdefault("relevance_score", 0.5)
            cleaned_result.setdefault("dataset_source", "unknown")
            cleaned_result.setdefault("language", "en")
            cleaned_result.setdefault("embedding", [])
            cleaned_result.setdefault("metadata", {})

            cleaned_results.append(cleaned_result)

        except Exception as e:
            # Skip problematic results but log the issue
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to clean search result: {str(e)}")
            continue

    return cleaned_results


def clean_cluster_data(clusters: List[Dict]) -> List[Dict]:
    """
    Clean cluster data for JSON serialization

    Args:
        clusters: List of cluster dictionaries

    Returns:
        Cleaned cluster data
    """
    cleaned_clusters = []

    for cluster in clusters:
        try:
            cleaned_cluster = clean_for_json(cluster)

            # Ensure required fields exist with defaults
            cleaned_cluster.setdefault("id", len(cleaned_clusters))
            cleaned_cluster.setdefault("label", "Unnamed Cluster")
            cleaned_cluster.setdefault("results", [])
            cleaned_cluster.setdefault("size", len(cleaned_cluster.get("results", [])))
            cleaned_cluster.setdefault("coherence_score", 0.5)
            cleaned_cluster.setdefault("diversity_score", 0.5)
            cleaned_cluster.setdefault("sources", [])
            cleaned_cluster.setdefault("categories", [])

            # Clean nested results
            if "results" in cleaned_cluster:
                cleaned_cluster["results"] = clean_search_results(
                    cleaned_cluster["results"]
                )

            cleaned_clusters.append(cleaned_cluster)

        except Exception as e:
            # Skip problematic clusters but log the issue
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to clean cluster data: {str(e)}")
            continue

    return cleaned_clusters


def clean_metrics_data(metrics: Dict) -> Dict:
    """
    Clean metrics data for JSON serialization

    Args:
        metrics: Metrics dictionary

    Returns:
        Cleaned metrics data
    """
    try:
        cleaned_metrics = clean_for_json(metrics)

        # Ensure numeric values are properly typed
        for key, value in cleaned_metrics.items():
            if key.endswith("_score") or key.endswith("_rate") or key.endswith("_pct"):
                try:
                    cleaned_metrics[key] = float(value) if value is not None else 0.0
                except (TypeError, ValueError):
                    cleaned_metrics[key] = 0.0

            elif (
                key.endswith("_count")
                or key.endswith("_total")
                or key in ["episodes", "queries"]
            ):
                try:
                    cleaned_metrics[key] = int(value) if value is not None else 0
                except (TypeError, ValueError):
                    cleaned_metrics[key] = 0

        return cleaned_metrics

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Failed to clean metrics data: {str(e)}")

        # Return minimal safe metrics
        return {
            "error": "Metrics cleaning failed",
            "total_queries": 0,
            "total_feedback": 0,
            "user_satisfaction_pct": 50,
        }


def validate_json_serializable(data: Any) -> bool:
    """
    Validate that data can be JSON serialized

    Args:
        data: Data to validate

    Returns:
        True if serializable, False otherwise
    """
    try:
        json.dumps(clean_for_json(data), cls=NumpyEncoder)
        return True
    except Exception:
        return False


def get_serialization_info(data: Any) -> Dict[str, Any]:
    """
    Get information about data serialization capabilities

    Args:
        data: Data to analyze

    Returns:
        Info dictionary
    """
    info = {
        "type": str(type(data).__name__),
        "is_serializable": False,
        "size_estimate": 0,
        "contains_numpy": False,
        "contains_datetime": False,
        "error": None,
    }

    try:
        # Check serializability
        cleaned_data = clean_for_json(data)
        json_str = json.dumps(cleaned_data, cls=NumpyEncoder)

        info["is_serializable"] = True
        info["size_estimate"] = len(json_str)

        # Check for specific types
        info["contains_numpy"] = _contains_numpy(data)
        info["contains_datetime"] = _contains_datetime(data)

    except Exception as e:
        info["error"] = str(e)

    return info


def _contains_numpy(data: Any, max_depth: int = 10) -> bool:
    """Check if data structure contains numpy objects"""
    if max_depth <= 0:
        return False

    if isinstance(data, (np.ndarray, np.floating, np.integer, np.bool_)):
        return True

    elif isinstance(data, dict):
        return any(_contains_numpy(value, max_depth - 1) for value in data.values())

    elif isinstance(data, (list, tuple)):
        return any(_contains_numpy(item, max_depth - 1) for item in data)

    return False


def _contains_datetime(data: Any, max_depth: int = 10) -> bool:
    """Check if data structure contains datetime objects"""
    if max_depth <= 0:
        return False

    if isinstance(data, datetime):
        return True

    elif isinstance(data, dict):
        return any(_contains_datetime(value, max_depth - 1) for value in data.values())

    elif isinstance(data, (list, tuple)):
        return any(_contains_datetime(item, max_depth - 1) for item in data)

    return False
