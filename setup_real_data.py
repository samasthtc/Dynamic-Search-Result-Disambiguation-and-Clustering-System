#!/usr/bin/env python3
"""
Real Data Setup Script
Downloads and processes real data from Wikipedia, ArXiv, and other sources
Limits to ~10 results per ambiguous term for manageable dataset size
"""

import sys
import os
import requests
import time
import json
import sqlite3
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataCollector:
    """Collects real data from various sources with limited results per term"""

    def __init__(self, data_dir="datasets", max_results_per_term=10):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_results_per_term = max_results_per_term

        # Initialize database
        self.db_path = self.data_dir / "real_search_data.db"
        self.init_database()

        # Initialize sentence transformer
        try:
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer: {e}")
            self.sentence_model = None

        # Curated list of ambiguous terms for real data collection
        self.ambiguous_terms = {
            "en": [
                "apple",  # Company vs fruit
                "python",  # Programming language vs snake
                "java",  # Programming language vs island/coffee
                "mercury",  # Planet vs element vs god
                "mars",  # Planet vs company vs god
                "amazon",  # Company vs river vs warriors
                "oracle",  # Company vs ancient oracle
                "jackson",  # Person vs place
                "paris",  # City vs person
                "phoenix",  # City vs mythical bird
                "eclipse",  # Astronomical vs software
                "jaguar",  # Animal vs car brand
                "windows",  # OS vs building feature
                "chrome",  # Browser vs metal
                "tiger",  # Animal vs golf player
            ],
            "ar": [
                "عين",  # Eye vs spring vs spy
                "بنك",  # Bank vs river bank
                "ورد",  # Rose vs mentioned
                "سلم",  # Peace vs ladder
                "نور",  # Light vs name
                "قلم",  # Pen vs region
                "جوز",  # Nuts vs husband
                "باب",  # Door vs chapter
                "نجم",  # Star vs celebrity
                "بحر",  # Sea vs meter (poetry)
            ],
        }

    def init_database(self):
        """Initialize SQLite database for real data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
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
                embedding TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

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

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_query_lang ON search_results(query, language)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON search_results(dataset_source)"
        )

        conn.commit()
        conn.close()
        logger.info(f"📊 Database initialized at {self.db_path}")

    def collect_all_real_data(self):
        """Collect real data from all sources"""
        logger.info("🚀 Starting real data collection...")

        total_collected = 0

        # Collect from Wikipedia
        logger.info("📚 Collecting Wikipedia data...")
        wiki_count = self.collect_wikipedia_data()
        total_collected += wiki_count

        # Collect from ArXiv (for technical terms)
        logger.info("🔬 Collecting ArXiv data...")
        arxiv_count = self.collect_arxiv_data()
        total_collected += arxiv_count

        # Generate statistics
        self.generate_statistics()

        logger.info(f"✅ Total collected: {total_collected} real results")
        return total_collected

    def collect_wikipedia_data(self):
        """Collect real Wikipedia data for ambiguous terms"""
        collected_count = 0

        for language in ["en", "ar"]:
            for term in self.ambiguous_terms[language]:
                try:
                    logger.info(f"🔍 Processing {term} ({language})...")

                    # Get Wikipedia disambiguation page
                    disambig_results = self.fetch_wikipedia_disambiguation(
                        term, language
                    )
                    collected_count += len(disambig_results)

                    # Get main Wikipedia page
                    main_results = self.fetch_wikipedia_main_page(term, language)
                    collected_count += len(main_results)

                    # Get Wikipedia search results
                    search_results = self.fetch_wikipedia_search(term, language)
                    collected_count += len(search_results)

                    # Rate limiting
                    time.sleep(1)

                    # Limit total results per term
                    if collected_count >= self.max_results_per_term:
                        break

                except Exception as e:
                    logger.warning(f"⚠️  Error processing {term}: {e}")
                    continue

        return collected_count

    def fetch_wikipedia_disambiguation(self, term, language):
        """Fetch Wikipedia disambiguation page"""
        try:
            wiki_base = f"https://{language}.wikipedia.org/api/rest_v1/"
            disambig_url = f"{wiki_base}page/summary/{term}_(disambiguation)"

            response = requests.get(disambig_url, timeout=10)
            if response.status_code != 200:
                return []

            data = response.json()

            # Store ambiguous query
            self.store_ambiguous_query(
                {
                    "query": term,
                    "language": language,
                    "ambiguity_level": 0.9,
                    "entity_types": ["multiple"],
                    "dataset_source": "wikipedia_disambiguation",
                    "num_meanings": 5,
                }
            )

            # Store search result
            result = self.create_search_result(
                query=term,
                language=language,
                title=data["title"],
                snippet=data.get("extract", "")[:500],
                url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                domain="wikipedia.org",
                category="disambiguation",
                source="wikipedia_disambiguation",
                relevance=0.95,
            )

            if result:
                self.store_search_result(result)
                return [result]

        except Exception as e:
            logger.debug(f"No disambiguation page for {term}: {e}")

        return []

    def fetch_wikipedia_main_page(self, term, language):
        """Fetch Wikipedia main page"""
        try:
            wiki_base = f"https://{language}.wikipedia.org/api/rest_v1/"
            main_url = f"{wiki_base}page/summary/{term}"

            response = requests.get(main_url, timeout=10)
            if response.status_code != 200:
                return []

            data = response.json()

            result = self.create_search_result(
                query=term,
                language=language,
                title=data["title"],
                snippet=data.get("extract", "")[:500],
                url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                domain="wikipedia.org",
                category=self.categorize_wikipedia_page(data),
                source="wikipedia_main",
                relevance=0.90,
            )

            if result:
                self.store_search_result(result)
                return [result]

        except Exception as e:
            logger.debug(f"No main page for {term}: {e}")

        return []

    def fetch_wikipedia_search(self, term, language):
        """Fetch Wikipedia search results"""
        results = []
        try:
            search_url = f"https://{language}.wikipedia.org/api/rest_v1/page/search"
            response = requests.get(
                search_url, params={"q": term, "limit": 5}, timeout=10
            )

            if response.status_code != 200:
                return []

            search_data = response.json()

            for item in search_data.get("pages", [])[:3]:  # Limit to 3 search results
                try:
                    # Get summary for each result
                    summary_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{item['key']}"
                    summary_response = requests.get(summary_url, timeout=10)

                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()

                        result = self.create_search_result(
                            query=term,
                            language=language,
                            title=summary_data["title"],
                            snippet=summary_data.get("extract", "")[:500],
                            url=summary_data.get("content_urls", {})
                            .get("desktop", {})
                            .get("page", ""),
                            domain="wikipedia.org",
                            category=self.categorize_wikipedia_page(summary_data),
                            source="wikipedia_search",
                            relevance=0.80,
                        )

                        if result:
                            self.store_search_result(result)
                            results.append(result)

                    time.sleep(0.5)  # Rate limiting

                except Exception as e:
                    logger.debug(f"Error processing search result: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Wikipedia search error for {term}: {e}")

        return results

    def collect_arxiv_data(self):
        """Collect ArXiv data for technical terms"""
        technical_terms = [
            "python",
            "java",
            "machine learning",
            "neural network",
            "algorithm",
        ]
        collected_count = 0

        for term in technical_terms:
            try:
                logger.info(f"🔬 Fetching ArXiv data for: {term}")

                import urllib.parse

                encoded_query = urllib.parse.quote(term)
                url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&max_results=3"

                response = requests.get(url, timeout=15)
                if response.status_code != 200:
                    continue

                # Parse XML response
                import xml.etree.ElementTree as ET

                root = ET.fromstring(response.content)

                for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                    title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                    summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                    link_elem = entry.find("{http://www.w3.org/2005/Atom}id")

                    if title_elem is not None and summary_elem is not None:
                        result = self.create_search_result(
                            query=term,
                            language="en",
                            title=title_elem.text.strip(),
                            snippet=summary_elem.text.strip()[:500],
                            url=link_elem.text if link_elem is not None else "",
                            domain="arxiv.org",
                            category="academic_paper",
                            source="arxiv_api",
                            relevance=0.85,
                        )

                        if result:
                            self.store_search_result(result)
                            collected_count += 1

                time.sleep(2)  # Rate limiting for ArXiv

            except Exception as e:
                logger.warning(f"ArXiv error for {term}: {e}")
                continue

        return collected_count

    def create_search_result(
        self, query, language, title, snippet, url, domain, category, source, relevance
    ):
        """Create a search result with embedding"""
        try:
            # Generate embedding if model available
            embedding = []
            if self.sentence_model:
                text = f"{title} {snippet}"
                embedding = self.sentence_model.encode(text).tolist()

            result = {
                "id": self.generate_id(f"{query}_{title}_{source}"),
                "query": query,
                "language": language,
                "title": title,
                "snippet": snippet,
                "url": url,
                "domain": domain,
                "category": category,
                "dataset_source": source,
                "relevance_score": relevance,
                "embedding": embedding,
                "metadata": {"collected_at": time.time(), "source": source},
            }

            return result

        except Exception as e:
            logger.error(f"Error creating search result: {e}")
            return None

    def store_search_result(self, result):
        """Store search result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO search_results 
                (id, query, language, title, snippet, url, domain, category,
                 dataset_source, relevance_score, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result["id"],
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
                    json.dumps(result["metadata"]),
                ),
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error storing result: {e}")
            return False

    def store_ambiguous_query(self, query_data):
        """Store ambiguous query in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query_id = self.generate_id(
                f"{query_data['query']}_{query_data['language']}"
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO ambiguous_queries
                (id, query, language, ambiguity_level, entity_types, 
                 dataset_source, num_meanings)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    query_id,
                    query_data["query"],
                    query_data["language"],
                    query_data["ambiguity_level"],
                    json.dumps(query_data["entity_types"]),
                    query_data["dataset_source"],
                    query_data["num_meanings"],
                ),
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error storing ambiguous query: {e}")
            return False

    def categorize_wikipedia_page(self, data):
        """Categorize Wikipedia page based on content"""
        title = data.get("title", "").lower()
        extract = data.get("extract", "").lower()

        if any(
            word in extract[:200] for word in ["born", "died", "politician", "actor"]
        ):
            return "person"
        elif any(word in extract[:200] for word in ["city", "country", "located"]):
            return "location"
        elif any(word in extract[:200] for word in ["company", "corporation"]):
            return "company"
        elif any(word in extract[:200] for word in ["software", "programming"]):
            return "technology"
        elif any(word in extract[:200] for word in ["animal", "species"]):
            return "animal"
        else:
            return "general"

    def generate_id(self, content):
        """Generate consistent ID from content"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]

    def generate_statistics(self):
        """Generate and display statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Count total results
            cursor.execute("SELECT COUNT(*) FROM search_results")
            total_results = cursor.fetchone()[0]

            # Count by source
            cursor.execute(
                "SELECT dataset_source, COUNT(*) FROM search_results GROUP BY dataset_source"
            )
            source_counts = dict(cursor.fetchall())

            # Count by language
            cursor.execute(
                "SELECT language, COUNT(*) FROM search_results GROUP BY language"
            )
            language_counts = dict(cursor.fetchall())

            # Count ambiguous queries
            cursor.execute("SELECT COUNT(*) FROM ambiguous_queries")
            ambiguous_count = cursor.fetchone()[0]

            conn.close()

            logger.info("📊 REAL DATA COLLECTION STATISTICS:")
            logger.info("=" * 50)
            logger.info(f"Total Results: {total_results}")
            logger.info(f"Ambiguous Queries: {ambiguous_count}")
            logger.info(f"Languages: {language_counts}")
            logger.info(f"Sources: {source_counts}")

            return {
                "total_results": total_results,
                "ambiguous_queries": ambiguous_count,
                "language_counts": language_counts,
                "source_counts": source_counts,
            }

        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {}


def main():
    """Main function to collect real data"""
    print("🚀 Real Data Collection Starting...")
    print("This will download real data from Wikipedia and ArXiv")
    print("Limited to ~10 results per term for manageable size\n")

    # Create data collector
    collector = RealDataCollector(max_results_per_term=10)

    # Collect all real data
    total_collected = collector.collect_all_real_data()

    if total_collected > 0:
        print(f"\n✅ SUCCESS! Collected {total_collected} real results")
        print(f"📊 Database: {collector.db_path}")
        print("\n🚀 Next steps:")
        print("1. Update your app.py to use the new database")
        print("2. Run: python app.py")
        print("3. Open: http://localhost:5000")
    else:
        print("\n❌ No data collected. Check your internet connection and try again.")


if __name__ == "__main__":
    main()
