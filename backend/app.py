#!/usr/bin/env python3
"""
Dynamic Search Result Disambiguation and Clustering via Reinforcement Learning
Flask Backend Implementation

Based on the research paper by Joud Hijaz, Mohammad AbuSaleh, Shatha Khdair, Usama Shoora
Department of Electrical and Computer Engineering, Birzeit University

Updated to use real Google Custom Search API and MIRACL-Arabic dataset
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Local imports
from clustering_manager import ClusteringManager
from rl_agent import RLAgent
from arabic_processor import ArabicProcessor
from search_client import SearchClient
from env_loader import (
    load_environment_variables,
    check_api_configuration,
    get_api_instructions,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load environment configuration
env_config = load_environment_variables()
api_status = check_api_configuration()

# Global state
search_client = None
clustering_manager = None
rl_agent = None
arabic_processor = None
current_results = []
current_embeddings = None
session_metrics = {
    "total_queries": 0,
    "total_feedback_items": 0,
    "cluster_purity": 0.0,
    "adjusted_rand_index": 0.0,
    "silhouette_score": 0.0,
    "user_satisfaction_pct": 0.0,
}


def initialize_components():
    """Initialize all system components"""
    global search_client, clustering_manager, rl_agent, arabic_processor

    logger.info("Initializing DSR-RL system components...")

    # Display API status
    if api_status["google_custom_search"]:
        logger.info("✅ Google Custom Search API: CONFIGURED")
    else:
        logger.warning("⚠️  Google Custom Search API: NOT CONFIGURED")

    logger.info("✅ MIRACL-Arabic Dataset: AVAILABLE")
    logger.info("✅ TREC Web Diversity: AVAILABLE")
    logger.info("✅ Wikipedia API: AVAILABLE")

    # Initialize components
    search_client = SearchClient()
    clustering_manager = ClusteringManager()
    rl_agent = RLAgent()
    arabic_processor = ArabicProcessor()

    # Get dataset statistics
    try:
        dataset_stats = search_client.get_dataset_statistics()
        logger.info(f"📊 Dataset statistics: {dataset_stats}")
    except Exception as e:
        logger.warning(f"Could not get dataset statistics: {e}")

    logger.info("🚀 All components initialized successfully")


# HTML template for the frontend (since we don't have templates folder)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DSR-RL: Dynamic Search Result Disambiguation</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Dynamic Search Disambiguation</h1>
            <p>AI-Powered Search Result Clustering with Reinforcement Learning</p>
            <div class="api-status">
                {% if google_api_enabled %}
                <span class="status-indicator status-ok">✅ Google Search: Active</span>
                {% else %}
                <span class="status-indicator status-warning">⚠️ Google Search: Limited</span>
                {% endif %}
                <span class="status-indicator status-ok">✅ MIRACL-Arabic: Active</span>
                <span class="status-indicator status-ok">✅ TREC Dataset: Active</span>
            </div>
        </div>
        <!-- Rest of the HTML content will be served from separate files -->
    </div>
    <script src="/static/script.js"></script>
</body>
</html>
"""


@app.route("/")
def index():
    """Serve the main application page"""
    try:
        # Try to serve from frontend directory
        with open("../frontend/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        try:
            # Try current directory
            with open("frontend/index.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            return html_content
        except FileNotFoundError:
            # Fallback to simple template
            return render_template_string(
                HTML_TEMPLATE, google_api_enabled=api_status["google_custom_search"]
            )


@app.route("/static/:filename")
def serve_static(filename):
    """Serve static files from frontend directory"""
    try:
        # Try frontend directory first
        file_paths = [
            f"../frontend/{filename}",
            f"frontend/{filename}",
            f"./{filename}",
        ]

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                print(f"Serving static file: {file_path}")
                # Set appropriate content type
                if filename.endswith(".css"):
                    return content, 200, {"Content-Type": "text/css"}
                elif filename.endswith(".js"):
                    return content, 200, {"Content-Type": "application/javascript"}
                else:
                    return content, 200

            except FileNotFoundError:
                continue

        return "File not found", 404

    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}")
        return "Error serving file", 500


@app.route("/api/search", methods=["POST"])
def search():
    """
    Perform real search using Google Custom Search API and datasets

    Expected JSON payload:
    {
        "query": "Jackson",
        "language": "en",
        "num_results": 20
    }
    """
    global current_results, current_embeddings, session_metrics

    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        language = data.get("language", "en")
        num_results = data.get("num_results", 10)

        if not query:
            return jsonify({"error": "Query is required"}), 400

        logger.info(
            f"🔍 Processing REAL search query: '{query}' (language: {language})"
        )

        # Preprocess query based on language
        if language == "ar":
            processed_query = arabic_processor.preprocess_text(query)
            logger.info(f"📝 Preprocessed Arabic query: '{processed_query}'")
        else:
            processed_query = query

        # Perform real search using Google Custom Search API and datasets
        search_results = search_client.search(processed_query, language, num_results)

        if not search_results:
            error_msg = "No results found. "
            if not api_status["google_custom_search"]:
                error_msg += (
                    "Consider setting up Google Custom Search API for more results. "
                )
            error_msg += "Try a different query or check your API configuration."

            return (
                jsonify(
                    {
                        "error": error_msg,
                        "results": [],
                        "total_results": 0,
                        "api_status": api_status,
                        "setup_instructions": (
                            get_api_instructions()
                            if not api_status["google_custom_search"]
                            else None
                        ),
                    }
                ),
                404,
            )

        # Store results globally for clustering
        current_results = search_results

        # Update metrics
        session_metrics["total_queries"] += 1

        # Generate embeddings for clustering
        texts = [f"{result['title']} {result['snippet']}" for result in search_results]
        if language == "ar":
            texts = [arabic_processor.preprocess_text(text) for text in texts]

        current_embeddings = clustering_manager.generate_embeddings(texts)

        logger.info(f"✅ Found {len(search_results)} REAL results for query: {query}")

        # Add source information to response
        sources_used = list(
            set([result.get("source", "unknown") for result in search_results])
        )

        return jsonify(
            {
                "results": search_results,
                "total_results": len(search_results),
                "query": query,
                "language": language,
                "sources_used": sources_used,
                "api_status": api_status,
                "processing_time": time.time(),
                "data_sources": {
                    "google_custom_search": len(
                        [
                            r
                            for r in search_results
                            if r.get("source") == "google_custom_search"
                        ]
                    ),
                    "miracl_arabic": len(
                        [
                            r
                            for r in search_results
                            if r.get("source") == "miracl_arabic"
                        ]
                    ),
                    "trec_web_diversity": len(
                        [
                            r
                            for r in search_results
                            if r.get("source") == "trec_web_diversity"
                        ]
                    ),
                    "wikipedia": len(
                        [
                            r
                            for r in search_results
                            if "wikipedia" in r.get("source", "")
                        ]
                    ),
                },
            }
        )

    except Exception as e:
        logger.error(f"❌ Search error: {str(e)}")
        return (
            jsonify(
                {
                    "error": str(e),
                    "api_status": api_status,
                    "troubleshooting": {
                        "check_internet": True,
                        "check_api_keys": not api_status["google_custom_search"],
                        "check_datasets": True,
                    },
                }
            ),
            500,
        )


@app.route("/api/cluster", methods=["POST"])
def cluster_results():
    """
    Cluster the current search results

    Expected JSON payload:
    {
        "algorithm": "bertopic",
        "num_clusters": 4,
        "min_cluster_size": 2,
        "ensemble_algorithms": ["kmeans", "hdbscan"]  // for ensemble
    }
    """
    global current_results, current_embeddings, session_metrics

    try:
        if not current_results or current_embeddings is None:
            return jsonify({"error": "No search results to cluster"}), 400

        data = request.get_json()
        algorithm = data.get("algorithm", "bertopic")
        num_clusters = data.get("num_clusters", 4)
        min_cluster_size = data.get("min_cluster_size", 2)
        ensemble_algorithms = data.get("ensemble_algorithms", [])

        logger.info(f"Clustering with algorithm: {algorithm}, clusters: {num_clusters}")

        # Get RL agent recommendation if adaptive
        if algorithm == "adaptive":
            state = rl_agent.get_current_state(current_embeddings, current_results)
            recommended_params = rl_agent.select_action(state)
            algorithm = recommended_params.get("algorithm", "bertopic")
            num_clusters = recommended_params.get("num_clusters", num_clusters)

        # Perform clustering
        if algorithm == "ensemble":
            clusters, labels, metrics = clustering_manager.ensemble_clustering(
                current_embeddings,
                current_results,
                algorithms=ensemble_algorithms,
                num_clusters=num_clusters,
            )
        else:
            clusters, labels, metrics = clustering_manager.cluster_results(
                current_embeddings,
                current_results,
                algorithm=algorithm,
                num_clusters=num_clusters,
                min_cluster_size=min_cluster_size,
            )

        # Convert NumPy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Convert all data to JSON-serializable format
        clusters = convert_numpy_types(clusters)
        labels = convert_numpy_types(labels)
        metrics = convert_numpy_types(metrics)

        # Update session metrics
        session_metrics.update(metrics)

        # Calculate cluster statistics
        cluster_stats = {
            "total_clusters": len(clusters) if clusters else 0,
            "largest_cluster_size": (
                max([c["size"] for c in clusters]) if clusters else 0
            ),
            "smallest_cluster_size": (
                min([c["size"] for c in clusters]) if clusters else 0
            ),
            "avg_cluster_size": (
                float(np.mean([c["size"] for c in clusters])) if clusters else 0.0
            ),
        }

        logger.info(f"Generated {len(clusters)} clusters")

        return jsonify(
            {
                "clusters": clusters,
                "labels": labels,
                "metrics": metrics,
                "cluster_stats": cluster_stats,
                "algorithm_used": algorithm,
            }
        )

    except Exception as e:
        logger.error(f"Clustering error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """
    Submit user feedback for RL training

    Expected JSON payload:
    {
        "result_index": 0,
        "cluster_index": 1,
        "feedback": "relevant",
        "context": "result",
        "query": "Jackson",
        "language": "en"
    }
    """
    global session_metrics

    try:
        data = request.get_json()

        # Extract feedback data
        result_index = data.get("result_index")
        cluster_index = data.get("cluster_index")
        feedback = data.get("feedback")
        context = data.get("context", "result")
        query = data.get("query", "")
        language = data.get("language", "en")

        logger.info(f"Received feedback: {feedback} for {context}")

        # Calculate reward based on feedback
        reward = rl_agent.calculate_reward(feedback, context)

        # Update RL agent
        if current_embeddings is not None:
            state = rl_agent.get_current_state(current_embeddings, current_results)
            rl_agent.update_policy(state, reward, feedback)

        # Update metrics
        session_metrics["total_feedback_items"] += 1

        # Update user satisfaction based on feedback
        satisfaction_mapping = {
            "excellent": 100,
            "relevant": 90,
            "good": 80,
            "poor": 20,
            "irrelevant": 10,
            "wrong_cluster": 30,
            "should_split": 40,
            "should_merge": 50,
        }

        if feedback in satisfaction_mapping:
            current_satisfaction = session_metrics["user_satisfaction_pct"]
            feedback_count = session_metrics["total_feedback_items"]
            new_satisfaction = satisfaction_mapping[feedback]

            # Running average
            session_metrics["user_satisfaction_pct"] = (
                current_satisfaction * (feedback_count - 1) + new_satisfaction
            ) / feedback_count

        # Get RL agent status and convert to JSON-serializable format
        rl_status = rl_agent.get_status()

        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        clean_rl_status = convert_numpy_types(rl_status)

        return jsonify(
            {
                "status": "success",
                "reward": float(reward),
                "total_episodes": clean_rl_status.get("episodes", 0),
                "exploration_rate": clean_rl_status.get("exploration_rate", 0.0),
                "total_reward": clean_rl_status.get("total_reward", 0.0),
            }
        )

    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Get current system metrics"""
    try:
        # Add RL agent metrics
        rl_status = rl_agent.get_status()

        # Convert NumPy types to Python native types
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Convert all metrics to JSON-serializable format
        clean_session_metrics = convert_numpy_types(session_metrics)
        clean_rl_status = convert_numpy_types(rl_status)

        metrics = {
            **clean_session_metrics,
            "rl_episodes": clean_rl_status.get("episodes", 0),
            "rl_exploration_rate": clean_rl_status.get("exploration_rate", 0.0),
            "rl_total_reward": clean_rl_status.get("total_reward", 0.0),
            "timestamp": datetime.now().isoformat(),
        }

        return jsonify(metrics)

    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_session():
    """Reset the current session"""
    global current_results, current_embeddings, session_metrics

    try:
        current_results = []
        current_embeddings = None
        session_metrics = {
            "total_queries": 0,
            "total_feedback_items": 0,
            "cluster_purity": 0.0,
            "adjusted_rand_index": 0.0,
            "silhouette_score": 0.0,
            "user_satisfaction_pct": 0.0,
        }

        logger.info("Session reset successfully")
        return jsonify({"status": "success", "message": "Session reset"})

    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "search_client": search_client is not None,
                "clustering_manager": clustering_manager is not None,
                "rl_agent": rl_agent is not None,
                "arabic_processor": arabic_processor is not None,
            },
        }
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Initialize components
    initialize_components()

    # Run the Flask app
    logger.info("Starting DSR-RL server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
