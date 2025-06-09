"""
Real Search System Core
Main orchestrator for real dataset-based search disambiguation
"""

import logging
import pickle
import json
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

from .datasets import DatasetManager
from .clustering import ClusteringEngine
from .feedback import FeedbackProcessor
from .json_utils import clean_for_json

logger = logging.getLogger(__name__)


class RealSearchSystem:
    """
    Core system that orchestrates real dataset search and clustering
    """

    def __init__(self):
        logger.info("Initializing Real Search System...")

        # Initialize components
        self.dataset_manager = DatasetManager()
        self.clustering_engine = ClusteringEngine()
        self.feedback_processor = FeedbackProcessor()

        # System state
        self.current_results = []
        self.current_clusters = []
        self.last_search_sources = []
        self.query_history = []

        # Load saved state if available
        self.load_state()

        logger.info("Real Search System initialized successfully")

    def search(
        self, query: str, language: str = "en", num_results: int = 20
    ) -> List[Dict]:
        """
        Perform search using real datasets

        Args:
            query: Search query
            language: Language code ('en' or 'ar')
            num_results: Number of results to return

        Returns:
            List of real search results
        """
        logger.info(f"Searching for '{query}' in {language}")

        try:
            # Get results from dataset manager
            results = self.dataset_manager.search(query, language, num_results)

            # Track sources used
            self.last_search_sources = list(
                set(result.get("dataset_source", "unknown") for result in results)
            )

            # Store current results
            self.current_results = results

            # Add to query history
            self.query_history.append(
                {
                    "query": query,
                    "language": language,
                    "timestamp": datetime.now().isoformat(),
                    "num_results": len(results),
                    "sources": self.last_search_sources,
                }
            )

            # Clean for JSON serialization
            return clean_for_json(results)

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    def cluster(
        self,
        algorithm: str = "kmeans",
        num_clusters: int = 4,
        min_cluster_size: int = 2,
    ) -> List[Dict]:
        """
        Cluster current search results

        Args:
            algorithm: Clustering algorithm to use
            num_clusters: Number of clusters
            min_cluster_size: Minimum cluster size

        Returns:
            List of clusters
        """
        if not self.current_results:
            logger.warning("No search results to cluster")
            return []

        logger.info(f"Clustering {len(self.current_results)} results with {algorithm}")

        try:
            # Perform clustering
            clusters = self.clustering_engine.cluster(
                self.current_results, algorithm, num_clusters, min_cluster_size
            )

            # Apply feedback-based optimization
            optimized_clusters = self.feedback_processor.optimize_clusters(
                clusters, self.get_recent_feedback()
            )

            self.current_clusters = optimized_clusters

            # Clean for JSON serialization
            return clean_for_json(optimized_clusters)

        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            return []

    def process_feedback(self, feedback_data: Dict) -> Dict:
        """
        Process user feedback and update system

        Args:
            feedback_data: User feedback information

        Returns:
            Processing result
        """
        try:
            # Add metadata
            feedback_data["timestamp"] = datetime.now().isoformat()
            feedback_data["query"] = (
                self.query_history[-1]["query"] if self.query_history else ""
            )

            # Process through feedback processor
            result = self.feedback_processor.process_feedback(feedback_data)

            return clean_for_json(result)

        except Exception as e:
            logger.error(f"Feedback processing error: {str(e)}")
            return {"status": "error", "error": str(e), "reward": 0.0}

    def get_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        try:
            # Get clustering metrics
            clustering_metrics = {}
            if self.current_clusters and self.current_results:
                clustering_metrics = self.clustering_engine.calculate_metrics(
                    self.current_results, self.current_clusters
                )

            # Get feedback metrics
            feedback_metrics = self.feedback_processor.get_metrics()

            # Get dataset metrics
            dataset_metrics = self.dataset_manager.get_statistics()

            # Combine all metrics
            metrics = {
                **clustering_metrics,
                **feedback_metrics,
                **dataset_metrics,
                "total_queries": len(self.query_history),
                "current_results_count": len(self.current_results),
                "current_clusters_count": len(self.current_clusters),
                "last_search_sources": self.last_search_sources,
            }

            return clean_for_json(metrics)

        except Exception as e:
            logger.error(f"Metrics calculation error: {str(e)}")
            return {"error": str(e)}

    def get_dataset_info(self) -> Dict:
        """Get information about loaded datasets"""
        try:
            return clean_for_json(self.dataset_manager.get_detailed_info())
        except Exception as e:
            logger.error(f"Dataset info error: {str(e)}")
            return {"error": str(e)}

    def get_ambiguous_queries(
        self, language: str = "en", limit: int = 20
    ) -> List[Dict]:
        """Get real ambiguous queries from datasets"""
        try:
            queries = self.dataset_manager.get_ambiguous_queries(language, limit)
            return clean_for_json(queries)
        except Exception as e:
            logger.error(f"Ambiguous queries error: {str(e)}")
            return []

    def get_loaded_datasets(self) -> List[str]:
        """Get list of successfully loaded datasets"""
        try:
            return self.dataset_manager.get_loaded_sources()
        except Exception as e:
            logger.error(f"Loaded datasets error: {str(e)}")
            return []

    def get_last_search_sources(self) -> List[str]:
        """Get sources used in last search"""
        return self.last_search_sources

    def get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        """Get recent feedback for optimization"""
        return self.feedback_processor.get_recent_feedback(limit)

    def save_state(self):
        """Save system state to disk"""
        try:
            # Save feedback processor state (includes RL agent)
            self.feedback_processor.save_state()

            # Save system state
            system_state = {
                "query_history": self.query_history[-100:],  # Keep last 100 queries
                "last_search_sources": self.last_search_sources,
                "timestamp": datetime.now().isoformat(),
            }

            with open("real_search_system_state.json", "w") as f:
                json.dump(system_state, f, indent=2)

            logger.debug("System state saved successfully")

        except Exception as e:
            logger.error(f"Save state error: {str(e)}")

    def load_state(self):
        """Load system state from disk"""
        try:
            # Load feedback processor state
            self.feedback_processor.load_state()

            # Load system state
            try:
                with open("real_search_system_state.json", "r") as f:
                    system_state = json.load(f)

                self.query_history = system_state.get("query_history", [])
                self.last_search_sources = system_state.get("last_search_sources", [])

                logger.info("System state loaded successfully")

            except FileNotFoundError:
                logger.info("No previous system state found, starting fresh")

        except Exception as e:
            logger.error(f"Load state error: {str(e)}")

    def reset_system(self):
        """Reset system to initial state"""
        try:
            self.current_results = []
            self.current_clusters = []
            self.last_search_sources = []
            self.query_history = []

            self.feedback_processor.reset()

            logger.info("System reset completed")

        except Exception as e:
            logger.error(f"System reset error: {str(e)}")

    def get_system_stats(self) -> Dict:
        """Get basic system statistics"""
        return {
            "total_queries": len(self.query_history),
            "current_results": len(self.current_results),
            "current_clusters": len(self.current_clusters),
            "available_datasets": len(self.dataset_manager.get_loaded_sources()),
            "feedback_items": self.feedback_processor.get_feedback_count(),
            "last_search_sources": self.last_search_sources,
        }
