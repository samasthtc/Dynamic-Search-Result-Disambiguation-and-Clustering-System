"""
Database Handler for Dataset Management
Handles SQLite database operations for caching and storing real dataset results
"""

import sqlite3
import json
import pickle
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """
    Manages SQLite database for storing and caching real dataset results
    """

    def __init__(self, db_path: str = "datasets/search_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)

        self.init_database()
        logger.info(f"Database handler initialized: {self.db_path}")

    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Search results table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS search_results (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                language TEXT NOT NULL,
                title TEXT NOT NULL,
                snippet TEXT,
                url TEXT,
                domain TEXT,
                category TEXT,
                dataset_source TEXT,
                relevance_score REAL DEFAULT 0.5,
                embedding BLOB,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Ambiguous queries table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ambiguous_queries (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                language TEXT NOT NULL,
                ambiguity_level REAL DEFAULT 0.5,
                entity_types TEXT,
                dataset_source TEXT,
                num_meanings INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Query-result relevance mappings (qrels)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS query_relevance (
                query_id TEXT,
                result_id TEXT,
                relevance_score REAL,
                dataset_source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (query_id, result_id, dataset_source)
            )
        """
        )

        # Indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_query_lang ON search_results(query, language)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relevance ON search_results(relevance_score)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON search_results(dataset_source)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_ambiguous_query ON ambiguous_queries(query, language)"
        )

        conn.commit()
        conn.close()

    def store_search_result(self, result: Dict[str, Any]) -> bool:
        """Store a search result in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Generate ID if not provided
            if "id" not in result:
                result["id"] = self._generate_id(
                    result.get("url", result.get("title", ""))
                )

            # Serialize embedding if present
            embedding_blob = None
            if "embedding" in result and result["embedding"] is not None:
                if isinstance(result["embedding"], (list, np.ndarray)):
                    embedding_blob = pickle.dumps(np.array(result["embedding"]))

            # Serialize metadata
            metadata_json = json.dumps(result.get("metadata", {}))

            cursor.execute(
                """
                INSERT OR REPLACE INTO search_results 
                (id, query, language, title, snippet, url, domain, category,
                 dataset_source, relevance_score, embedding, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result["id"],
                    result.get("query", ""),
                    result.get("language", "en"),
                    result.get("title", ""),
                    result.get("snippet", ""),
                    result.get("url", ""),
                    result.get("domain", ""),
                    result.get("category", "general"),
                    result.get("dataset_source", "unknown"),
                    result.get("relevance_score", 0.5),
                    embedding_blob,
                    metadata_json,
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error storing search result: {str(e)}")
            return False

    def store_ambiguous_query(self, query_data: Dict[str, Any]) -> bool:
        """Store an ambiguous query in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Generate ID if not provided
            if "id" not in query_data:
                query_data["id"] = self._generate_id(
                    f"{query_data.get('query', '')}_{query_data.get('language', 'en')}"
                )

            # Serialize entity types
            entity_types_json = json.dumps(query_data.get("entity_types", []))

            cursor.execute(
                """
                INSERT OR REPLACE INTO ambiguous_queries
                (id, query, language, ambiguity_level, entity_types, 
                 dataset_source, num_meanings)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    query_data["id"],
                    query_data.get("query", ""),
                    query_data.get("language", "en"),
                    query_data.get("ambiguity_level", 0.5),
                    entity_types_json,
                    query_data.get("dataset_source", "unknown"),
                    query_data.get("num_meanings", 1),
                ),
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error storing ambiguous query: {str(e)}")
            return False

    def store_relevance_mapping(
        self, query_id: str, result_id: str, relevance: float, source: str
    ) -> bool:
        """Store query-result relevance mapping"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO query_relevance
                (query_id, result_id, relevance_score, dataset_source)
                VALUES (?, ?, ?, ?)
            """,
                (query_id, result_id, relevance, source),
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error storing relevance mapping: {str(e)}")
            return False

    def get_search_results(
        self, query: str, language: str = "en", limit: int = 20, source: str = None
    ) -> List[Dict[str, Any]]:
        """Retrieve search results from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build query based on parameters
            if source:
                cursor.execute(
                    """
                    SELECT * FROM search_results 
                    WHERE (query = ? OR title LIKE ? OR snippet LIKE ?) 
                    AND language = ? AND dataset_source = ?
                    ORDER BY relevance_score DESC
                    LIMIT ?
                """,
                    (query, f"%{query}%", f"%{query}%", language, source, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM search_results 
                    WHERE (query = ? OR title LIKE ? OR snippet LIKE ?) 
                    AND language = ?
                    ORDER BY relevance_score DESC
                    LIMIT ?
                """,
                    (query, f"%{query}%", f"%{query}%", language, limit),
                )

            results = cursor.fetchall()
            conn.close()

            # Convert to dictionaries
            formatted_results = []
            for row in results:
                try:
                    # Deserialize embedding
                    embedding = None
                    if row[10]:  # embedding column
                        embedding = pickle.loads(row[10]).tolist()

                    # Deserialize metadata
                    metadata = {}
                    if row[11]:  # metadata column
                        metadata = json.loads(row[11])

                    result = {
                        "id": row[0],
                        "query": row[1],
                        "language": row[2],
                        "title": row[3],
                        "snippet": row[4] or "",
                        "url": row[5] or "",
                        "domain": row[6] or "",
                        "category": row[7] or "general",
                        "dataset_source": row[8] or "unknown",
                        "relevance_score": row[9] or 0.5,
                        "embedding": embedding or [],
                        "metadata": metadata,
                        "created_at": row[12],
                        "updated_at": row[13],
                        "final_score": row[9] or 0.5,  # Use relevance as final score
                    }

                    formatted_results.append(result)

                except Exception as e:
                    logger.warning(f"Error processing result row: {str(e)}")
                    continue

            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving search results: {str(e)}")
            return []

    def get_ambiguous_queries(
        self, language: str = "en", limit: int = 50, source: str = None
    ) -> List[Dict[str, Any]]:
        """Retrieve ambiguous queries from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if source:
                cursor.execute(
                    """
                    SELECT * FROM ambiguous_queries 
                    WHERE language = ? AND dataset_source = ?
                    ORDER BY ambiguity_level DESC
                    LIMIT ?
                """,
                    (language, source, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM ambiguous_queries 
                    WHERE language = ?
                    ORDER BY ambiguity_level DESC
                    LIMIT ?
                """,
                    (language, limit),
                )

            results = cursor.fetchall()
            conn.close()

            # Convert to dictionaries
            queries = []
            for row in results:
                try:
                    entity_types = json.loads(row[4]) if row[4] else []

                    query = {
                        "id": row[0],
                        "query": row[1],
                        "language": row[2],
                        "ambiguity_level": row[3],
                        "entity_types": entity_types,
                        "dataset_source": row[5],
                        "num_meanings": row[6],
                        "created_at": row[7],
                    }

                    queries.append(query)

                except Exception as e:
                    logger.warning(f"Error processing query row: {str(e)}")
                    continue

            return queries

        except Exception as e:
            logger.error(f"Error retrieving ambiguous queries: {str(e)}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Count total results
            cursor.execute("SELECT COUNT(*) FROM search_results")
            total_results = cursor.fetchone()[0]

            # Count by language
            cursor.execute(
                "SELECT language, COUNT(*) FROM search_results GROUP BY language"
            )
            lang_counts = dict(cursor.fetchall())

            # Count by source
            cursor.execute(
                "SELECT dataset_source, COUNT(*) FROM search_results GROUP BY dataset_source"
            )
            source_counts = dict(cursor.fetchall())

            # Count ambiguous queries
            cursor.execute("SELECT COUNT(*) FROM ambiguous_queries")
            ambiguous_count = cursor.fetchone()[0]

            # Count relevance mappings
            cursor.execute("SELECT COUNT(*) FROM query_relevance")
            relevance_count = cursor.fetchone()[0]

            conn.close()

            return {
                "total_results": total_results,
                "ambiguous_queries": ambiguous_count,
                "relevance_mappings": relevance_count,
                "results_by_language": lang_counts,
                "results_by_source": source_counts,
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}

    def clear_source_data(self, source: str) -> bool:
        """Clear all data from a specific source"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM search_results WHERE dataset_source = ?", (source,)
            )
            cursor.execute(
                "DELETE FROM ambiguous_queries WHERE dataset_source = ?", (source,)
            )
            cursor.execute(
                "DELETE FROM query_relevance WHERE dataset_source = ?", (source,)
            )

            conn.commit()
            conn.close()

            logger.info(f"Cleared data for source: {source}")
            return True

        except Exception as e:
            logger.error(f"Error clearing source data: {str(e)}")
            return False

    def _generate_id(self, content: str) -> str:
        """Generate consistent ID from content"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
