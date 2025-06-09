"""
Wikipedia Dataset Source
Handles Wikipedia API for real disambiguation pages and articles
"""

import requests
import time
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from ..core import DatasetSource
from ..database import DatabaseHandler

logger = logging.getLogger(__name__)


class WikipediaSource(DatasetSource):
    """
    Wikipedia dataset source for real disambiguation data
    """

    def __init__(self, db_handler: DatabaseHandler, rate_limit: float = 1.0):
        self.db_handler = db_handler
        self.rate_limit = rate_limit  # seconds between requests
        self.last_request_time = 0

        # Initialize sentence transformer for embeddings
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Known ambiguous terms for different languages
        self.ambiguous_terms = {
            "en": [
                "apple",
                "python",
                "java",
                "mercury",
                "mars",
                "amazon",
                "oracle",
                "jackson",
                "washington",
                "victoria",
                "cambridge",
                "oxford",
                "jaguar",
                "tiger",
                "phoenix",
                "aurora",
                "eclipse",
                "windows",
                "chrome",
                "safari",
                "firefox",
                "opera",
                "paris",
                "berlin",
                "rome",
                "athens",
                "dublin",
                "warsaw",
                "vienna",
                "brussels",
            ],
            "ar": [
                "عين",
                "بنك",
                "ورد",
                "سلم",
                "نور",
                "قلم",
                "باب",
                "كتاب",
                "جوز",
                "مفتاح",
                "حديد",
                "ذهب",
                "فضة",
                "نجم",
                "قمر",
                "شمس",
                "بحر",
                "جبل",
                "نهر",
                "واد",
                "صحراء",
            ],
        }

        self.loaded = False
        logger.info("Wikipedia source initialized")

    def _rate_limit_wait(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()

    def load_data(self) -> bool:
        """Load Wikipedia disambiguation data"""
        logger.info("Loading Wikipedia disambiguation data...")

        success_count = 0
        total_count = 0

        for language in ["en", "ar"]:
            for term in self.ambiguous_terms[language]:
                total_count += 1
                try:
                    if self._load_term_data(term, language):
                        success_count += 1
                    time.sleep(0.1)  # Be nice to Wikipedia
                except Exception as e:
                    logger.warning(f"Failed to load {term} ({language}): {str(e)}")

        self.loaded = success_count > 0
        logger.info(f"Loaded {success_count}/{total_count} Wikipedia terms")
        return self.loaded

    def _load_term_data(self, term: str, language: str) -> bool:
        """Load data for a specific term"""
        self._rate_limit_wait()

        wiki_base = f"https://{language}.wikipedia.org/api/rest_v1/"

        # Try to get disambiguation page first
        success = False

        # Check disambiguation page
        if self._fetch_disambiguation_page(term, language, wiki_base):
            success = True

        # Get main page
        if self._fetch_main_page(term, language, wiki_base):
            success = True

        return success

    def _fetch_disambiguation_page(
        self, term: str, language: str, wiki_base: str
    ) -> bool:
        """Fetch disambiguation page for term"""
        try:
            disambig_url = f"{wiki_base}page/summary/{term}_(disambiguation)"
            response = requests.get(disambig_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return self._process_disambiguation_page(term, language, data)

        except Exception as e:
            logger.debug(f"No disambiguation page for {term}: {str(e)}")

        return False

    def _fetch_main_page(self, term: str, language: str, wiki_base: str) -> bool:
        """Fetch main Wikipedia page for term"""
        try:
            main_url = f"{wiki_base}page/summary/{term}"
            response = requests.get(main_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return self._process_main_page(term, language, data)

        except Exception as e:
            logger.debug(f"No main page for {term}: {str(e)}")

        return False

    def _process_disambiguation_page(
        self, term: str, language: str, data: Dict
    ) -> bool:
        """Process disambiguation page data"""
        if "extract" not in data:
            return False

        extract = data["extract"]

        # Count potential meanings
        meaning_indicators = ["may refer to", "refers to", "disambiguation", "can mean"]
        num_meanings = sum(
            extract.lower().count(indicator) for indicator in meaning_indicators
        )
        num_meanings = max(num_meanings, 2)  # At least 2 for disambiguation pages

        # Store ambiguous query
        query_data = {
            "query": term,
            "language": language,
            "ambiguity_level": min(1.0, num_meanings / 10.0),
            "entity_types": ["multiple"],
            "dataset_source": "wikipedia_disambiguation",
            "num_meanings": num_meanings,
        }

        self.db_handler.store_ambiguous_query(query_data)

        # Store search result
        result_data = {
            "query": term,
            "language": language,
            "title": data["title"],
            "snippet": extract[:500],
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "domain": "wikipedia.org",
            "category": "disambiguation",
            "dataset_source": "wikipedia_disambiguation",
            "relevance_score": 0.9,
            "metadata": {
                "page_id": data.get("pageid"),
                "type": "disambiguation",
                "num_meanings": num_meanings,
            },
        }

        # Generate embedding
        text_for_embedding = f"{data['title']} {extract}"
        embedding = self.sentence_model.encode(text_for_embedding)
        result_data["embedding"] = embedding.tolist()

        return self.db_handler.store_search_result(result_data)

    def _process_main_page(self, term: str, language: str, data: Dict) -> bool:
        """Process main Wikipedia page data"""
        if "extract" not in data:
            return False

        extract = data["extract"]

        # Determine category
        category = self._categorize_page(data)

        # Store search result
        result_data = {
            "query": term,
            "language": language,
            "title": data["title"],
            "snippet": extract[:500],
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "domain": "wikipedia.org",
            "category": category,
            "dataset_source": "wikipedia_main",
            "relevance_score": 0.85,
            "metadata": {"page_id": data.get("pageid"), "type": "main_article"},
        }

        # Generate embedding
        text_for_embedding = f"{data['title']} {extract}"
        embedding = self.sentence_model.encode(text_for_embedding)
        result_data["embedding"] = embedding.tolist()

        return self.db_handler.store_search_result(result_data)

    def _categorize_page(self, data: Dict) -> str:
        """Categorize Wikipedia page based on content"""
        title = data.get("title", "").lower()
        extract = data.get("extract", "").lower()

        # Category detection based on content
        if any(word in title for word in ["company", "corporation", "inc", "ltd"]):
            return "company"
        elif any(
            word in extract[:200]
            for word in ["born", "died", "politician", "actor", "singer", "musician"]
        ):
            return "person"
        elif any(
            word in extract[:200]
            for word in ["city", "country", "located", "geography", "capital"]
        ):
            return "location"
        elif any(
            word in extract[:200]
            for word in ["software", "programming", "technology", "computer"]
        ):
            return "technology"
        elif any(
            word in extract[:200]
            for word in ["animal", "species", "genus", "bird", "mammal"]
        ):
            return "animal"
        elif any(
            word in extract[:200] for word in ["planet", "star", "astronomy", "space"]
        ):
            return "astronomy"
        elif any(
            word in extract[:200]
            for word in ["element", "chemical", "chemistry", "compound"]
        ):
            return "chemistry"
        else:
            return "general"

    def get_results(
        self, query: str, language: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get search results for query"""
        results = self.db_handler.get_search_results(
            query, language, limit, source="wikipedia_disambiguation"
        )

        # If no disambiguation results, get main page results
        if not results:
            results = self.db_handler.get_search_results(
                query, language, limit, source="wikipedia_main"
            )

        # If still no results, try live API fetch
        if not results and self.loaded:
            self._fetch_live_results(query, language, limit)
            results = self.db_handler.get_search_results(
                query, language, limit, source="wikipedia_live"
            )

        return results

    def _fetch_live_results(self, query: str, language: str, limit: int) -> bool:
        """Fetch live results from Wikipedia API"""
        try:
            self._rate_limit_wait()

            # Search for articles
            search_url = f"https://{language}.wikipedia.org/api/rest_v1/page/search"
            response = requests.get(
                search_url, params={"q": query, "limit": min(limit, 5)}, timeout=10
            )

            if response.status_code != 200:
                return False

            search_results = response.json()

            for item in search_results.get("pages", []):
                self._rate_limit_wait()

                # Get full article summary
                summary_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{item['key']}"
                summary_response = requests.get(summary_url, timeout=10)

                if summary_response.status_code == 200:
                    summary_data = summary_response.json()

                    result_data = {
                        "query": query,
                        "language": language,
                        "title": summary_data["title"],
                        "snippet": summary_data.get("extract", "")[:500],
                        "url": summary_data.get("content_urls", {})
                        .get("desktop", {})
                        .get("page", ""),
                        "domain": "wikipedia.org",
                        "category": self._categorize_page(summary_data),
                        "dataset_source": "wikipedia_live",
                        "relevance_score": 0.8,
                        "metadata": {
                            "page_id": summary_data.get("pageid"),
                            "type": "live_search",
                        },
                    }

                    # Generate embedding
                    text_for_embedding = (
                        f"{summary_data['title']} {summary_data.get('extract', '')}"
                    )
                    embedding = self.sentence_model.encode(text_for_embedding)
                    result_data["embedding"] = embedding.tolist()

                    self.db_handler.store_search_result(result_data)

            return True

        except Exception as e:
            logger.error(f"Error fetching live Wikipedia results: {str(e)}")
            return False

    def get_ambiguous_queries(self, language: str, limit: int) -> List[Dict[str, Any]]:
        """Get ambiguous queries"""
        return self.db_handler.get_ambiguous_queries(
            language, limit, source="wikipedia_disambiguation"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for this source"""
        stats = self.db_handler.get_statistics()

        # Filter for Wikipedia sources
        wikipedia_sources = [
            "wikipedia_disambiguation",
            "wikipedia_main",
            "wikipedia_live",
        ]
        wikipedia_results = sum(
            stats.get("results_by_source", {}).get(source, 0)
            for source in wikipedia_sources
        )

        return {
            "initialized": self.loaded,
            "total_results": wikipedia_results,
            "disambiguation_pages": stats.get("results_by_source", {}).get(
                "wikipedia_disambiguation", 0
            ),
            "main_pages": stats.get("results_by_source", {}).get("wikipedia_main", 0),
            "live_results": stats.get("results_by_source", {}).get("wikipedia_live", 0),
            "supported_languages": ["en", "ar"],
            "ambiguous_terms_count": {
                "en": len(self.ambiguous_terms["en"]),
                "ar": len(self.ambiguous_terms["ar"]),
            },
        }
