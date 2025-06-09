"""
Core Dataset Manager
Main interface for managing real-world datasets
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DatasetSource(ABC):
    """Abstract base class for dataset sources"""

    @abstractmethod
    def load_data(self) -> bool:
        """Load data from source"""
        pass

    @abstractmethod
    def get_results(
        self, query: str, language: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get search results for query"""
        pass

    @abstractmethod
    def get_ambiguous_queries(self, language: str, limit: int) -> List[Dict[str, Any]]:
        """Get ambiguous queries"""
        pass


class DatasetManager:
    """
    Main dataset manager that coordinates multiple data sources
    """

    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.sources: Dict[str, DatasetSource] = {}
        self.initialized = False

        logger.info(f"Dataset Manager initialized with data directory: {self.data_dir}")

    def register_source(self, name: str, source: DatasetSource):
        """Register a new dataset source"""
        self.sources[name] = source
        logger.info(f"Registered dataset source: {name}")

    def initialize(self) -> bool:
        """Initialize all registered sources"""
        success_count = 0

        for name, source in self.sources.items():
            try:
                if source.load_data():
                    success_count += 1
                    logger.info(f"Successfully initialized {name}")
                else:
                    logger.warning(f"Failed to initialize {name}")
            except Exception as e:
                logger.error(f"Error initializing {name}: {str(e)}")

        self.initialized = success_count > 0
        logger.info(f"Initialized {success_count}/{len(self.sources)} dataset sources")

        return self.initialized

    def get_search_results(
        self, query: str, language: str = "en", num_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get search results from all available sources
        """
        if not self.initialized:
            logger.warning("Dataset manager not initialized")
            return []

        all_results = []

        for name, source in self.sources.items():
            try:
                results = source.get_results(query, language, num_results)
                for result in results:
                    result["source"] = name
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error getting results from {name}: {str(e)}")

        # Sort by relevance and limit
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return all_results[:num_results]

    def get_ambiguous_queries(
        self, language: str = "en", limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get ambiguous queries from all sources"""
        if not self.initialized:
            return []

        all_queries = []

        for name, source in self.sources.items():
            try:
                queries = source.get_ambiguous_queries(language, limit)
                for query in queries:
                    query["source"] = name
                all_queries.extend(queries)
            except Exception as e:
                logger.error(f"Error getting queries from {name}: {str(e)}")

        # Remove duplicates and sort by ambiguity level
        seen_queries = set()
        unique_queries = []

        for query in all_queries:
            query_key = (query["query"], query["language"])
            if query_key not in seen_queries:
                seen_queries.add(query_key)
                unique_queries.append(query)

        unique_queries.sort(key=lambda x: x.get("ambiguity_level", 0), reverse=True)
        return unique_queries[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from all sources"""
        stats = {
            "total_sources": len(self.sources),
            "initialized_sources": 0,
            "total_results": 0,
            "total_queries": 0,
            "sources": {},
        }

        for name, source in self.sources.items():
            try:
                if hasattr(source, "get_statistics"):
                    source_stats = source.get_statistics()
                    stats["sources"][name] = source_stats
                    stats["total_results"] += source_stats.get("total_results", 0)
                    stats["total_queries"] += source_stats.get("total_queries", 0)
                    if source_stats.get("initialized", False):
                        stats["initialized_sources"] += 1
            except Exception as e:
                logger.warning(f"Could not get statistics from {name}: {str(e)}")

        return stats
