# """
# Dataset Manager
# Handles all real dataset sources and provides unified search interface
# """

# import logging
# import requests
# import json
# import time
# import sqlite3
# import hashlib
# from pathlib import Path
# from typing import List, Dict, Any
# from sentence_transformers import SentenceTransformer

# logger = logging.getLogger(__name__)


# class DatasetManager:
#     """
#     Manages multiple real dataset sources and provides unified search interface
#     """

#     def __init__(self, data_dir: str = "datasets"):
#         self.data_dir = Path(data_dir)
#         self.data_dir.mkdir(exist_ok=True)

#         # Initialize database
#         self.db_path = self.data_dir / "search_cache.db"
#         self.init_database()

#         # Initialize sentence transformer
#         self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

#         # Track loaded sources
#         self.loaded_sources = set()

#         # Initialize all sources
#         self.init_sources()

#         logger.info(
#             f"Dataset Manager initialized with {len(self.loaded_sources)} sources"
#         )

#     def init_database(self):
#         """Initialize SQLite cache database"""
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()

#         # Cache table for search results
#         cursor.execute(
#             """
#             CREATE TABLE IF NOT EXISTS search_cache (
#                 id TEXT PRIMARY KEY,
#                 query TEXT,
#                 language TEXT,
#                 title TEXT,
#                 snippet TEXT,
#                 url TEXT,
#                 domain TEXT,
#                 category TEXT,
#                 dataset_source TEXT,
#                 relevance_score REAL,
#                 embedding TEXT,
#                 metadata TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#         """
#         )

#         # Ambiguous queries table
#         cursor.execute(
#             """
#             CREATE TABLE IF NOT EXISTS ambiguous_queries (
#                 id TEXT PRIMARY KEY,
#                 query TEXT,
#                 language TEXT,
#                 ambiguity_level REAL,
#                 entity_types TEXT,
#                 dataset_source TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#         """
#         )

#         cursor.execute(
#             "CREATE INDEX IF NOT EXISTS idx_query ON search_cache(query, language)"
#         )
#         cursor.execute(
#             "CREATE INDEX IF NOT EXISTS idx_source ON search_cache(dataset_source)"
#         )

#         conn.commit()
#         conn.close()

#     def init_sources(self):
#         """Initialize all dataset sources"""
#         # Wikipedia source
#         if self.init_wikipedia():
#             self.loaded_sources.add("wikipedia")

#         # Live academic APIs
#         if self.init_live_apis():
#             self.loaded_sources.add("live_apis")

#         # Sample fallback data
#         self.init_sample_data()
#         self.loaded_sources.add("sample_data")

#     def init_wikipedia(self) -> bool:
#         """Initialize Wikipedia source"""
#         try:
#             # Test Wikipedia API
#             response = requests.get(
#                 "https://en.wikipedia.org/api/rest_v1/page/summary/Python", timeout=5
#             )

#             if response.status_code == 200:
#                 # Load some common ambiguous terms
#                 self.load_wikipedia_terms()
#                 return True

#         except Exception as e:
#             logger.warning(f"Wikipedia initialization failed: {str(e)}")

#         return False

#     def load_wikipedia_terms(self):
#         """Load common ambiguous terms from Wikipedia"""
#         ambiguous_terms = {
#             "en": ["apple", "python", "java", "mercury", "jackson", "paris", "phoenix"],
#             "ar": ["عين", "بنك", "ورد", "سلم", "نور"],
#         }

#         for language, terms in ambiguous_terms.items():
#             for term in terms[:3]:  # Limit to avoid rate limiting
#                 try:
#                     self.fetch_wikipedia_data(term, language)
#                     time.sleep(0.5)  # Rate limiting
#                 except Exception as e:
#                     logger.debug(f"Error loading {term}: {str(e)}")

#     def fetch_wikipedia_data(self, term: str, language: str):
#         """Fetch data for a term from Wikipedia"""
#         wiki_base = f"https://{language}.wikipedia.org/api/rest_v1/"

#         try:
#             # Try main page
#             response = requests.get(f"{wiki_base}page/summary/{term}", timeout=10)

#             if response.status_code == 200:
#                 data = response.json()
#                 self.store_wikipedia_result(term, language, data, "main")

#             # Try disambiguation page
#             response = requests.get(
#                 f"{wiki_base}page/summary/{term}_(disambiguation)", timeout=10
#             )

#             if response.status_code == 200:
#                 data = response.json()
#                 self.store_wikipedia_result(term, language, data, "disambiguation")

#                 # Store as ambiguous query
#                 self.store_ambiguous_query(
#                     {
#                         "query": term,
#                         "language": language,
#                         "ambiguity_level": 0.8,
#                         "entity_types": ["multiple"],
#                         "dataset_source": "wikipedia",
#                     }
#                 )

#         except Exception as e:
#             logger.debug(f"Wikipedia fetch error for {term}: {str(e)}")

#     def store_wikipedia_result(
#         self, query: str, language: str, data: Dict, result_type: str
#     ):
#         """Store Wikipedia result in cache"""
#         if "extract" not in data:
#             return

#         # Generate embedding
#         text = f"{data['title']} {data['extract']}"
#         embedding = self.sentence_model.encode(text)

#         result = {
#             "query": query,
#             "language": language,
#             "title": data["title"],
#             "snippet": data["extract"][:500],
#             "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
#             "domain": "wikipedia.org",
#             "category": self.categorize_wikipedia(data),
#             "dataset_source": f"wikipedia_{result_type}",
#             "relevance_score": 0.9 if result_type == "main" else 0.85,
#             "embedding": embedding.tolist(),
#             "metadata": {"page_id": data.get("pageid"), "type": result_type},
#         }

#         self.store_result(result)

#     def categorize_wikipedia(self, data: Dict) -> str:
#         """Categorize Wikipedia page"""
#         title = data.get("title", "").lower()
#         extract = data.get("extract", "").lower()

#         if any(
#             word in extract[:200] for word in ["born", "died", "politician", "actor"]
#         ):
#             return "person"
#         elif any(word in extract[:200] for word in ["city", "country", "located"]):
#             return "location"
#         elif any(word in extract[:200] for word in ["company", "corporation"]):
#             return "company"
#         elif any(word in extract[:200] for word in ["software", "programming"]):
#             return "technology"
#         else:
#             return "general"

#     def init_live_apis(self) -> bool:
#         """Initialize live API sources"""
#         try:
#             # Test ArXiv API
#             response = requests.get(
#                 "http://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results=1",
#                 timeout=5,
#             )

#             if response.status_code == 200:
#                 return True

#         except Exception as e:
#             logger.warning(f"Live APIs initialization failed: {str(e)}")

#         return False

#     def init_sample_data(self):
#         """Initialize sample data for demonstration"""
#         sample_queries = [
#             {
#                 "query": "machine learning",
#                 "language": "en",
#                 "title": "Introduction to Machine Learning",
#                 "snippet": "Machine learning is a method of data analysis that automates analytical model building.",
#                 "url": "https://example.com/ml-intro",
#                 "domain": "example.com",
#                 "category": "technology",
#                 "dataset_source": "sample_data",
#                 "relevance_score": 0.7,
#             },
#             {
#                 "query": "التعلم الآلي",
#                 "language": "ar",
#                 "title": "مقدمة في التعلم الآلي",
#                 "snippet": "التعلم الآلي هو طريقة لتحليل البيانات التي تؤتمت بناء النماذج التحليلية.",
#                 "url": "https://example.com/ml-intro-ar",
#                 "domain": "example.com",
#                 "category": "technology",
#                 "dataset_source": "sample_data",
#                 "relevance_score": 0.7,
#             },
#         ]

#         for sample in sample_queries:
#             # Generate embedding
#             text = f"{sample['title']} {sample['snippet']}"
#             embedding = self.sentence_model.encode(text)
#             sample["embedding"] = embedding.tolist()
#             sample["metadata"] = {"type": "sample"}

#             self.store_result(sample)

#     def store_result(self, result: Dict):
#         """Store result in cache database"""
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()

#             result_id = hashlib.md5(
#                 f"{result['query']}_{result['title']}_{result['dataset_source']}".encode()
#             ).hexdigest()

#             cursor.execute(
#                 """
#                 INSERT OR REPLACE INTO search_cache
#                 (id, query, language, title, snippet, url, domain, category,
#                  dataset_source, relevance_score, embedding, metadata)
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             """,
#                 (
#                     result_id,
#                     result["query"],
#                     result["language"],
#                     result["title"],
#                     result["snippet"],
#                     result["url"],
#                     result["domain"],
#                     result["category"],
#                     result["dataset_source"],
#                     result["relevance_score"],
#                     json.dumps(result["embedding"]),
#                     json.dumps(result["metadata"]),
#                 ),
#             )

#             conn.commit()
#             conn.close()

#         except Exception as e:
#             logger.error(f"Store result error: {str(e)}")

#     def store_ambiguous_query(self, query_data: Dict):
#         """Store ambiguous query"""
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()

#             query_id = hashlib.md5(
#                 f"{query_data['query']}_{query_data['language']}".encode()
#             ).hexdigest()

#             cursor.execute(
#                 """
#                 INSERT OR REPLACE INTO ambiguous_queries
#                 (id, query, language, ambiguity_level, entity_types, dataset_source)
#                 VALUES (?, ?, ?, ?, ?, ?)
#             """,
#                 (
#                     query_id,
#                     query_data["query"],
#                     query_data["language"],
#                     query_data["ambiguity_level"],
#                     json.dumps(query_data["entity_types"]),
#                     query_data["dataset_source"],
#                 ),
#             )

#             conn.commit()
#             conn.close()

#         except Exception as e:
#             logger.error(f"Store ambiguous query error: {str(e)}")

#     def search(self, query: str, language: str, num_results: int) -> List[Dict]:
#         """
#         Search across all dataset sources

#         Args:
#             query: Search query
#             language: Language code
#             num_results: Number of results to return

#         Returns:
#             List of search results
#         """
#         results = []

#         # Search cache first
#         cached_results = self.search_cache(query, language, num_results)
#         results.extend(cached_results)

#         # If not enough cached results, try live sources
#         if len(results) < num_results:
#             if "wikipedia" in self.loaded_sources:
#                 wiki_results = self.search_wikipedia_live(query, language, 5)
#                 results.extend(wiki_results)

#             if "live_apis" in self.loaded_sources and language == "en":
#                 api_results = self.search_live_apis(query, 5)
#                 results.extend(api_results)

#         # Sort by relevance and limit
#         results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
#         return results[:num_results]

#     def search_cache(self, query: str, language: str, limit: int) -> List[Dict]:
#         """Search cached results"""
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()

#             # Try exact match first
#             cursor.execute(
#                 """
#                 SELECT * FROM search_cache 
#                 WHERE query = ? AND language = ?
#                 ORDER BY relevance_score DESC
#                 LIMIT ?
#             """,
#                 (query, language, limit),
#             )

#             results = cursor.fetchall()

#             # If no exact match, try partial match
#             if not results:
#                 cursor.execute(
#                     """
#                     SELECT * FROM search_cache 
#                     WHERE (title LIKE ? OR snippet LIKE ?) AND language = ?
#                     ORDER BY relevance_score DESC
#                     LIMIT ?
#                 """,
#                     (f"%{query}%", f"%{query}%", language, limit),
#                 )

#                 results = cursor.fetchall()

#             conn.close()

#             # Convert to dictionaries
#             return [self.row_to_dict(row) for row in results]

#         except Exception as e:
#             logger.error(f"Cache search error: {str(e)}")
#             return []

#     def search_wikipedia_live(
#         self, query: str, language: str, limit: int
#     ) -> List[Dict]:
#         """Search Wikipedia live API"""
#         try:
#             search_url = f"https://{language}.wikipedia.org/api/rest_v1/page/search"
#             response = requests.get(
#                 search_url, params={"q": query, "limit": min(limit, 3)}, timeout=10
#             )

#             if response.status_code != 200:
#                 return []

#             search_results = response.json()
#             results = []

#             for item in search_results.get("pages", []):
#                 try:
#                     # Get summary for each result
#                     summary_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{item['key']}"
#                     summary_response = requests.get(summary_url, timeout=10)

#                     if summary_response.status_code == 200:
#                         data = summary_response.json()

#                         # Generate embedding
#                         text = f"{data['title']} {data.get('extract', '')}"
#                         embedding = self.sentence_model.encode(text)

#                         result = {
#                             "id": f"wiki_live_{item['key']}",
#                             "query": query,
#                             "language": language,
#                             "title": data["title"],
#                             "snippet": data.get("extract", "")[:500],
#                             "url": data.get("content_urls", {})
#                             .get("desktop", {})
#                             .get("page", ""),
#                             "domain": "wikipedia.org",
#                             "category": self.categorize_wikipedia(data),
#                             "dataset_source": "wikipedia_live",
#                             "relevance_score": 0.8,
#                             "embedding": embedding.tolist(),
#                             "metadata": {
#                                 "page_id": data.get("pageid"),
#                                 "live_search": True,
#                             },
#                         }

#                         results.append(result)

#                         # Store in cache for future use
#                         self.store_result(result)

#                     time.sleep(0.2)  # Rate limiting

#                 except Exception as e:
#                     logger.debug(f"Error processing Wikipedia result: {str(e)}")
#                     continue

#             return results

#         except Exception as e:
#             logger.error(f"Wikipedia live search error: {str(e)}")
#             return []

#     def search_live_apis(self, query: str, limit: int) -> List[Dict]:
#         """Search live academic APIs"""
#         results = []

#         # Try ArXiv
#         try:
#             arxiv_results = self.search_arxiv(query, min(limit, 2))
#             results.extend(arxiv_results)
#         except Exception as e:
#             logger.debug(f"ArXiv search error: {str(e)}")

#         return results

#     def search_arxiv(self, query: str, limit: int) -> List[Dict]:
#         """Search ArXiv API"""
#         try:
#             import urllib.parse

#             encoded_query = urllib.parse.quote(query)
#             url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&max_results={limit}"

#             response = requests.get(url, timeout=15)
#             response.raise_for_status()

#             # Parse XML response
#             import xml.etree.ElementTree as ET

#             root = ET.fromstring(response.content)

#             results = []

#             for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
#                 title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
#                 summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
#                 link_elem = entry.find("{http://www.w3.org/2005/Atom}id")

#                 if title_elem is not None and summary_elem is not None:
#                     title = title_elem.text.strip()
#                     summary = summary_elem.text.strip()
#                     link = link_elem.text if link_elem is not None else ""

#                     # Generate embedding
#                     text = f"{title} {summary}"
#                     embedding = self.sentence_model.encode(text)

#                     result = {
#                         "id": f"arxiv_{hashlib.md5(link.encode()).hexdigest()[:8]}",
#                         "query": query,
#                         "language": "en",
#                         "title": title,
#                         "snippet": summary[:500],
#                         "url": link,
#                         "domain": "arxiv.org",
#                         "category": "academic_paper",
#                         "dataset_source": "arxiv_live",
#                         "relevance_score": 0.8,
#                         "embedding": embedding.tolist(),
#                         "metadata": {"source": "arxiv", "live_search": True},
#                     }

#                     results.append(result)

#                     # Store in cache
#                     self.store_result(result)

#             return results

#         except Exception as e:
#             logger.error(f"ArXiv search error: {str(e)}")
#             return []

#     def row_to_dict(self, row) -> Dict:
#         """Convert database row to dictionary"""
#         try:
#             return {
#                 "id": row[0],
#                 "query": row[1],
#                 "language": row[2],
#                 "title": row[3],
#                 "snippet": row[4],
#                 "url": row[5],
#                 "domain": row[6],
#                 "category": row[7],
#                 "dataset_source": row[8],
#                 "relevance_score": row[9],
#                 "embedding": json.loads(row[10]) if row[10] else [],
#                 "metadata": json.loads(row[11]) if row[11] else {},
#                 "created_at": row[12],
#             }
#         except Exception as e:
#             logger.error(f"Row conversion error: {str(e)}")
#             return {}

#     def get_ambiguous_queries(self, language: str, limit: int) -> List[Dict]:
#         """Get ambiguous queries from database"""
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()

#             cursor.execute(
#                 """
#                 SELECT * FROM ambiguous_queries 
#                 WHERE language = ?
#                 ORDER BY ambiguity_level DESC
#                 LIMIT ?
#             """,
#                 (language, limit),
#             )

#             results = cursor.fetchall()
#             conn.close()

#             queries = []
#             for row in results:
#                 queries.append(
#                     {
#                         "id": row[0],
#                         "query": row[1],
#                         "language": row[2],
#                         "ambiguity_level": row[3],
#                         "entity_types": json.loads(row[4]) if row[4] else [],
#                         "dataset_source": row[5],
#                         "created_at": row[6],
#                     }
#                 )

#             return queries

#         except Exception as e:
#             logger.error(f"Get ambiguous queries error: {str(e)}")
#             return []

#     def get_statistics(self) -> Dict:
#         """Get dataset statistics"""
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()

#             # Count total results
#             cursor.execute("SELECT COUNT(*) FROM search_cache")
#             total_results = cursor.fetchone()[0]

#             # Count by source
#             cursor.execute(
#                 "SELECT dataset_source, COUNT(*) FROM search_cache GROUP BY dataset_source"
#             )
#             source_counts = dict(cursor.fetchall())

#             # Count by language
#             cursor.execute(
#                 "SELECT language, COUNT(*) FROM search_cache GROUP BY language"
#             )
#             language_counts = dict(cursor.fetchall())

#             # Count ambiguous queries
#             cursor.execute("SELECT COUNT(*) FROM ambiguous_queries")
#             ambiguous_count = cursor.fetchone()[0]

#             conn.close()

#             return {
#                 "total_results": total_results,
#                 "results_by_source": source_counts,
#                 "results_by_language": language_counts,
#                 "ambiguous_queries": ambiguous_count,
#                 "loaded_sources": list(self.loaded_sources),
#             }

#         except Exception as e:
#             logger.error(f"Statistics error: {str(e)}")
#             return {}

#     def get_detailed_info(self) -> Dict:
#         """Get detailed dataset information"""
#         stats = self.get_statistics()

#         return {
#             "status": "active",
#             "loaded_sources": list(self.loaded_sources),
#             "statistics": stats,
#             "capabilities": {
#                 "wikipedia_search": "wikipedia" in self.loaded_sources,
#                 "live_api_search": "live_apis" in self.loaded_sources,
#                 "cached_results": stats.get("total_results", 0) > 0,
#                 "multilingual": "ar" in stats.get("results_by_language", {}),
#                 "ambiguous_queries": stats.get("ambiguous_queries", 0) > 0,
#             },
#             "data_directory": str(self.data_dir),
#             "cache_database": str(self.db_path),
#         }

#     def get_loaded_sources(self) -> List[str]:
#         """Get list of loaded sources"""
#         return list(self.loaded_sources)

#     def clear_cache(self):
#         """Clear all cached data"""
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()

#             cursor.execute("DELETE FROM search_cache")
#             cursor.execute("DELETE FROM ambiguous_queries")

#             conn.commit()
#             conn.close()

#             logger.info("Cache cleared successfully")

#         except Exception as e:
#             logger.error(f"Clear cache error: {str(e)}")

#     def refresh_data(self):
#         """Refresh data from all sources"""
#         try:
#             # Clear cache
#             self.clear_cache()

#             # Reinitialize sources
#             self.loaded_sources.clear()
#             self.init_sources()

#             logger.info("Data refresh completed")

#         except Exception as e:
#             logger.error(f"Data refresh error: {str(e)}")

"""
Dataset Manager - Updated for Real Data
Handles real dataset sources and provides unified search interface
"""

import logging
import requests
import json
import time
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages real dataset sources and provides unified search interface
    """

    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Use the real data database
        self.db_path = self.data_dir / "real_search_data.db"
        
        # Fallback to cache database if real data not available
        if not self.db_path.exists():
            self.db_path = self.data_dir / "search_cache.db"
            
        self.init_database()

        # Initialize sentence transformer
        try:
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.warning(f"⚠️  Sentence transformer not available: {e}")
            self.sentence_model = None

        # Track loaded sources
        self.loaded_sources = set()
        self.check_available_data()

        logger.info(f"Dataset Manager initialized with {len(self.loaded_sources)} sources")

    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Ensure tables exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_results (
                id TEXT PRIMARY KEY,
                query TEXT,
                language TEXT,
                title TEXT,
                snippet TEXT,
                url TEXT,
                domain TEXT,
                category TEXT,
                dataset_source TEXT,
                relevance_score REAL,
                embedding TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ambiguous_queries (
                id TEXT PRIMARY KEY,
                query TEXT,
                language TEXT,
                ambiguity_level REAL,
                entity_types TEXT,
                dataset_source TEXT,
                num_meanings INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query ON search_results(query, language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON search_results(dataset_source)")

        conn.commit()
        conn.close()

    def check_available_data(self):
        """Check what data sources are available"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check available sources
            cursor.execute("SELECT DISTINCT dataset_source FROM search_results")
            sources = [row[0] for row in cursor.fetchall()]
            
            # Check data counts
            cursor.execute("SELECT COUNT(*) FROM search_results")
            total_results = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ambiguous_queries")
            total_queries = cursor.fetchone()[0]
            
            conn.close()
            
            if total_results > 0:
                self.loaded_sources = set(sources)
                logger.info(f"📊 Found {total_results} results from {len(sources)} sources")
                logger.info(f"🔍 Found {total_queries} ambiguous queries")
            else:
                logger.warning("⚠️  No real data found, will create sample data")
                self.create_sample_data()
                self.loaded_sources.add("sample_data")
                
        except Exception as e:
            logger.error(f"Error checking available data: {e}")
            self.create_sample_data()
            self.loaded_sources.add("sample_data")

    def create_sample_data(self):
        """Create minimal sample data if no real data available"""
        sample_data = [
            {
                "query": "python",
                "language": "en",
                "title": "Python (programming language)",
                "snippet": "Python is a high-level programming language known for its simplicity and readability.",
                "url": "https://www.python.org",
                "domain": "python.org",
                "category": "technology",
                "dataset_source": "sample_data",
                "relevance_score": 0.9
            },
            {
                "query": "python",
                "language": "en", 
                "title": "Python (snake)",
                "snippet": "Pythons are large non-venomous snakes found in Africa, Asia, and Australia.",
                "url": "https://en.wikipedia.org/wiki/Python_(mythology)",
                "domain": "wikipedia.org",
                "category": "animal",
                "dataset_source": "sample_data",
                "relevance_score": 0.85
            },
            {
                "query": "apple",
                "language": "en",
                "title": "Apple Inc.",
                "snippet": "Apple Inc. is an American multinational technology company headquartered in Cupertino, California.",
                "url": "https://www.apple.com",
                "domain": "apple.com", 
                "category": "company",
                "dataset_source": "sample_data",
                "relevance_score": 0.9
            },
            {
                "query": "apple",
                "language": "en",
                "title": "Apple (fruit)",
                "snippet": "An apple is a sweet, edible fruit produced by an apple tree.",
                "url": "https://en.wikipedia.org/wiki/Apple",
                "domain": "wikipedia.org",
                "category": "food",
                "dataset_source": "sample_data", 
                "relevance_score": 0.85
            },
            {
                "query": "عين",
                "language": "ar",
                "title": "العين (عضو)",
                "snippet": "العين هي عضو الرؤية في الكائنات الحية.",
                "url": "https://ar.wikipedia.org/wiki/عين",
                "domain": "wikipedia.org",
                "category": "anatomy",
                "dataset_source": "sample_data",
                "relevance_score": 0.9
            },
            {
                "query": "عين",
                "language": "ar", 
                "title": "عين الماء",
                "snippet": "عين الماء هي مصدر طبيعي للمياه العذبة.",
                "url": "https://ar.wikipedia.org/wiki/عين_ماء",
                "domain": "wikipedia.org",
                "category": "geography",
                "dataset_source": "sample_data",
                "relevance_score": 0.85
            }
        ]
        
        # Store sample data
        for item in sample_data:
            self.store_result(item)
        
        # Store sample ambiguous queries
        ambiguous_queries = [
            {
                "query": "python",
                "language": "en",
                "ambiguity_level": 0.8,
                "entity_types": ["technology", "animal"],
                "dataset_source": "sample_data",
                "num_meanings": 2
            },
            {
                "query": "apple", 
                "language": "en",
                "ambiguity_level": 0.7,
                "entity_types": ["company", "food"],
                "dataset_source": "sample_data",
                "num_meanings": 2
            },
            {
                "query": "عين",
                "language": "ar",
                "ambiguity_level": 0.9,
                "entity_types": ["anatomy", "geography"],
                "dataset_source": "sample_data", 
                "num_meanings": 3
            }
        ]
        
        for query in ambiguous_queries:
            self.store_ambiguous_query(query)
        
        logger.info("📝 Created sample data")

    def search(self, query: str, language: str, num_results: int) -> List[Dict]:
        """
        Search for results in the database
        """
        results = []

        # First try exact query match
        exact_results = self.search_database(query, language, num_results)
        results.extend(exact_results)

        # If not enough results, try fuzzy matching
        if len(results) < num_results:
            fuzzy_results = self.search_database_fuzzy(query, language, num_results - len(results))
            # Avoid duplicates
            existing_ids = {r.get('id') for r in results}
            for result in fuzzy_results:
                if result.get('id') not in existing_ids:
                    results.append(result)

        # If still not enough and we have live capability, try fetching new data
        if len(results) < 3:
            live_results = self.fetch_live_data(query, language, 3)
            results.extend(live_results)

        # Sort by relevance and limit
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results[:num_results]

    def search_database(self, query: str, language: str, limit: int) -> List[Dict]:
        """Search database for exact query matches"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM search_results 
                WHERE query = ? AND language = ?
                ORDER BY relevance_score DESC
                LIMIT ?
            """, (query, language, limit))

            results = cursor.fetchall()
            conn.close()

            return [self.row_to_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Database search error: {e}")
            return []

    def search_database_fuzzy(self, query: str, language: str, limit: int) -> List[Dict]:
        """Search database with fuzzy matching"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM search_results 
                WHERE (title LIKE ? OR snippet LIKE ? OR query LIKE ?) 
                AND language = ?
                ORDER BY relevance_score DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", language, limit))

            results = cursor.fetchall()
            conn.close()

            return [self.row_to_dict(row) for row in results]

        except Exception as e:
            logger.error(f"Fuzzy search error: {e}")
            return []

    def fetch_live_data(self, query: str, language: str, limit: int) -> List[Dict]:
        """Fetch live data from Wikipedia if needed"""
        if language not in ["en", "ar"]:
            return []
            
        try:
            logger.info(f"🔍 Fetching live data for: {query}")
            
            # Try Wikipedia search
            wiki_base = f"https://{language}.wikipedia.org/api/rest_v1/"
            search_url = f"{wiki_base}page/search"
            
            response = requests.get(
                search_url,
                params={"q": query, "limit": min(limit, 3)},
                timeout=10
            )
            
            if response.status_code != 200:
                return []
                
            search_data = response.json()
            results = []
            
            for item in search_data.get("pages", [])[:limit]:
                try:
                    # Get page summary
                    summary_url = f"{wiki_base}page/summary/{item['key']}"
                    summary_response = requests.get(summary_url, timeout=10)
                    
                    if summary_response.status_code == 200:
                        data = summary_response.json()
                        
                        # Create result
                        result = {
                            "id": self.generate_id(f"live_{query}_{data['title']}"),
                            "query": query,
                            "language": language,
                            "title": data["title"],
                            "snippet": data.get("extract", "")[:500],
                            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                            "domain": "wikipedia.org",
                            "category": self.categorize_content(data),
                            "dataset_source": "wikipedia_live",
                            "relevance_score": 0.75,
                            "embedding": self.generate_embedding(f"{data['title']} {data.get('extract', '')}"),
                            "metadata": {"live_fetch": True}
                        }
                        
                        # Store for future use
                        self.store_result(result)
                        results.append(result)
                        
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Error processing live result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.warning(f"Live data fetch error: {e}")
            return []

    def store_result(self, result: Dict):
        """Store result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Generate embedding if not present
            if "embedding" not in result or not result["embedding"]:
                result["embedding"] = self.generate_embedding(f"{result.get('title', '')} {result.get('snippet', '')}")

            cursor.execute("""
                INSERT OR REPLACE INTO search_results
                (id, query, language, title, snippet, url, domain, category,
                 dataset_source, relevance_score, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.get("id", self.generate_id(f"{result['query']}_{result['title']}")),
                result["query"],
                result["language"],
                result["title"],
                result["snippet"],
                result["url"],
                result["domain"],
                result["category"],
                result["dataset_source"],
                result["relevance_score"],
                json.dumps(result["embedding"]),
                json.dumps(result.get("metadata", {}))
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error storing result: {e}")
            return False

    def store_ambiguous_query(self, query_data: Dict):
        """Store ambiguous query"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query_id = self.generate_id(f"{query_data['query']}_{query_data['language']}")

            cursor.execute("""
                INSERT OR REPLACE INTO ambiguous_queries
                (id, query, language, ambiguity_level, entity_types, dataset_source, num_meanings)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id,
                query_data["query"],
                query_data["language"],
                query_data["ambiguity_level"],
                json.dumps(query_data["entity_types"]),
                query_data["dataset_source"],
                query_data["num_meanings"]
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing ambiguous query: {e}")

    def row_to_dict(self, row) -> Dict:
        """Convert database row to dictionary"""
        try:
            embedding = []
            if row[10]:  # embedding column
                try:
                    embedding = json.loads(row[10])
                except:
                    embedding = []

            metadata = {}
            if row[11]:  # metadata column
                try:
                    metadata = json.loads(row[11])
                except:
                    metadata = {}

            return {
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
                "embedding": embedding,
                "metadata": metadata,
                "created_at": row[12] if len(row) > 12 else ""
            }
        except Exception as e:
            logger.error(f"Row conversion error: {e}")
            return {}

    def get_ambiguous_queries(self, language: str, limit: int) -> List[Dict]:
        """Get ambiguous queries from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM ambiguous_queries 
                WHERE language = ?
                ORDER BY ambiguity_level DESC
                LIMIT ?
            """, (language, limit))

            results = cursor.fetchall()
            conn.close()

            queries = []
            for row in results:
                try:
                    entity_types = json.loads(row[4]) if row[4] else []
                    queries.append({
                        "id": row[0],
                        "query": row[1],
                        "language": row[2],
                        "ambiguity_level": row[3],
                        "entity_types": entity_types,
                        "dataset_source": row[5],
                        "num_meanings": row[6],
                        "created_at": row[7] if len(row) > 7 else ""
                    })
                except Exception as e:
                    logger.debug(f"Error processing query row: {e}")
                    continue

            return queries

        except Exception as e:
            logger.error(f"Get ambiguous queries error: {e}")
            return []

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not self.sentence_model or not text:
            return []
        
        try:
            embedding = self.sentence_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.debug(f"Embedding generation error: {e}")
            return []

    def categorize_content(self, data: Dict) -> str:
        """Categorize content based on title and extract"""
        title = data.get("title", "").lower()
        extract = data.get("extract", "").lower()
        
        # Simple categorization rules
        if any(word in extract[:200] for word in ["born", "died", "politician", "actor", "singer"]):
            return "person"
        elif any(word in extract[:200] for word in ["city", "country", "located", "capital"]):
            return "location"
        elif any(word in extract[:200] for word in ["company", "corporation", "inc", "ltd"]):
            return "company"
        elif any(word in extract[:200] for word in ["software", "programming", "technology"]):
            return "technology"
        elif any(word in extract[:200] for word in ["animal", "species", "bird", "mammal"]):
            return "animal"
        elif any(word in extract[:200] for word in ["planet", "star", "astronomy"]):
            return "astronomy"
        else:
            return "general"

    def generate_id(self, content: str) -> str:
        """Generate consistent ID from content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Count total results
            cursor.execute("SELECT COUNT(*) FROM search_results")
            total_results = cursor.fetchone()[0]

            # Count by source
            cursor.execute("SELECT dataset_source, COUNT(*) FROM search_results GROUP BY dataset_source")
            source_counts = dict(cursor.fetchall())

            # Count by language
            cursor.execute("SELECT language, COUNT(*) FROM search_results GROUP BY language")
            language_counts = dict(cursor.fetchall())

            # Count ambiguous queries
            cursor.execute("SELECT COUNT(*) FROM ambiguous_queries")
            ambiguous_count = cursor.fetchone()[0]

            conn.close()

            return {
                "total_results": total_results,
                "results_by_source": source_counts,
                "results_by_language": language_counts,
                "ambiguous_queries": ambiguous_count,
                "loaded_sources": list(self.loaded_sources)
            }

        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return {}

    def get_detailed_info(self) -> Dict:
        """Get detailed dataset information"""
        stats = self.get_statistics()

        return {
            "status": "active",
            "data_type": "real_data" if "wikipedia" in str(self.loaded_sources) else "sample_data",
            "loaded_sources": list(self.loaded_sources),
            "statistics": stats,
            "capabilities": {
                "real_wikipedia_data": any("wikipedia" in source for source in self.loaded_sources),
                "live_fetching": True,
                "cached_results": stats.get("total_results", 0) > 0,
                "multilingual": len(stats.get("results_by_language", {})) > 1,
                "ambiguous_queries": stats.get("ambiguous_queries", 0) > 0,
            },
            "data_directory": str(self.data_dir),
            "database_path": str(self.db_path),
        }

    def get_loaded_sources(self) -> List[str]:
        """Get list of loaded sources"""
        return list(self.loaded_sources)