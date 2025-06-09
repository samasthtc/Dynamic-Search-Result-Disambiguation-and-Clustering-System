"""
Live API Sources
Handles real-time data from external APIs like ArXiv, CrossRef, etc.
"""

import requests
import time
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from ..core import DatasetSource
from ..database import DatabaseHandler

logger = logging.getLogger(__name__)


class LiveAPISource(DatasetSource):
    """
    Source for live API data from ArXiv, CrossRef, and other academic sources
    """

    def __init__(self, db_handler: DatabaseHandler, rate_limit: float = 2.0):
        self.db_handler = db_handler
        self.rate_limit = rate_limit
        self.last_request_time = 0

        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # API configurations
        self.apis = {
            "arxiv": {
                "base_url": "http://export.arxiv.org/api/query",
                "enabled": True,
                "rate_limit": 3.0,  # seconds
            },
            "crossref": {
                "base_url": "https://api.crossref.org/works",
                "enabled": True,
                "rate_limit": 1.0,
            },
            "semantic_scholar": {
                "base_url": "https://api.semanticscholar.org/graph/v1/paper/search",
                "enabled": True,
                "rate_limit": 1.0,
            },
        }

        self.loaded = False
        logger.info("Live API source initialized")

    def load_data(self) -> bool:
        """Load data - for live APIs, this just validates connectivity"""
        logger.info("Validating live API connectivity...")

        working_apis = 0

        for api_name, config in self.apis.items():
            if config["enabled"]:
                try:
                    if self._test_api_connectivity(api_name):
                        working_apis += 1
                        logger.info(f"{api_name} API is accessible")
                    else:
                        logger.warning(f"{api_name} API is not accessible")
                        config["enabled"] = False
                except Exception as e:
                    logger.warning(f"Error testing {api_name}: {str(e)}")
                    config["enabled"] = False

        self.loaded = working_apis > 0
        logger.info(f"{working_apis}/{len(self.apis)} live APIs are working")
        return self.loaded

    def _test_api_connectivity(self, api_name: str) -> bool:
        """Test if an API is accessible"""
        try:
            if api_name == "arxiv":
                response = requests.get(
                    "http://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results=1",
                    timeout=10,
                )
                return response.status_code == 200

            elif api_name == "crossref":
                response = requests.get(
                    "https://api.crossref.org/works?query=machine+learning&rows=1",
                    headers={"User-Agent": "SearchDisambiguation/1.0"},
                    timeout=10,
                )
                return response.status_code == 200

            elif api_name == "semantic_scholar":
                response = requests.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search?query=machine+learning&limit=1",
                    timeout=10,
                )
                return response.status_code == 200

        except Exception:
            return False

        return False

    def _rate_limit_wait(self, api_name: str):
        """Enforce rate limiting for specific API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        api_rate_limit = self.apis.get(api_name, {}).get("rate_limit", self.rate_limit)

        if time_since_last < api_rate_limit:
            time.sleep(api_rate_limit - time_since_last)

        self.last_request_time = time.time()

    def get_results(
        self, query: str, language: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get search results for query from live APIs"""
        if language != "en":
            return []  # Most academic APIs are English only

        all_results = []

        # Fetch from ArXiv
        if self.apis["arxiv"]["enabled"]:
            arxiv_results = self._fetch_arxiv_results(query, min(limit // 3, 5))
            all_results.extend(arxiv_results)

        # Fetch from CrossRef
        if self.apis["crossref"]["enabled"]:
            crossref_results = self._fetch_crossref_results(query, min(limit // 3, 5))
            all_results.extend(crossref_results)

        # Fetch from Semantic Scholar
        if self.apis["semantic_scholar"]["enabled"]:
            ss_results = self._fetch_semantic_scholar_results(query, min(limit // 3, 5))
            all_results.extend(ss_results)

        # Sort by relevance and limit
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return all_results[:limit]

    def _fetch_arxiv_results(
        self, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """Fetch results from ArXiv API"""
        try:
            self._rate_limit_wait("arxiv")

            import urllib.parse

            encoded_query = urllib.parse.quote(query)
            url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&max_results={max_results}"

            response = requests.get(url, timeout=15)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)
            results = []

            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                link_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                published_elem = entry.find("{http://www.w3.org/2005/Atom}published")

                if title_elem is not None and summary_elem is not None:
                    title = title_elem.text.strip()
                    summary = summary_elem.text.strip()
                    link = link_elem.text if link_elem is not None else ""
                    published = (
                        published_elem.text if published_elem is not None else ""
                    )

                    result_data = {
                        "query": query,
                        "language": "en",
                        "title": title,
                        "snippet": summary[:500],
                        "url": link,
                        "domain": "arxiv.org",
                        "category": "academic_paper",
                        "dataset_source": "arxiv_live_api",
                        "relevance_score": 0.8,
                        "metadata": {
                            "source": "arxiv",
                            "published": published,
                            "api_source": True,
                        },
                    }

                    # Generate embedding
                    text_for_embedding = f"{title} {summary}"
                    embedding = self.sentence_model.encode(text_for_embedding)
                    result_data["embedding"] = embedding.tolist()

                    # Store in database
                    self.db_handler.store_search_result(result_data)
                    results.append(result_data)

            logger.info(f"Fetched {len(results)} results from ArXiv")
            return results

        except Exception as e:
            logger.error(f"Error fetching ArXiv results: {str(e)}")
            return []

    def _fetch_crossref_results(
        self, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """Fetch results from CrossRef API"""
        try:
            self._rate_limit_wait("crossref")

            params = {
                "query": query,
                "rows": max_results,
                "select": "title,abstract,DOI,URL,author,published-print,type",
            }

            response = requests.get(
                "https://api.crossref.org/works",
                params=params,
                headers={"User-Agent": "SearchDisambiguation/1.0"},
                timeout=15,
            )
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("message", {}).get("items", []):
                title_list = item.get("title", [])
                title = title_list[0] if title_list else "Untitled"

                abstract_list = item.get("abstract", [])
                abstract = abstract_list[0] if abstract_list else ""

                doi = item.get("DOI", "")
                url = item.get("URL", f"https://doi.org/{doi}" if doi else "")
                doc_type = item.get("type", "article")

                # Get publication date
                pub_date = ""
                if "published-print" in item:
                    date_parts = item["published-print"].get("date-parts", [[]])
                    if date_parts and date_parts[0]:
                        pub_date = (
                            f"{date_parts[0][0]}" if len(date_parts[0]) >= 1 else ""
                        )

                result_data = {
                    "query": query,
                    "language": "en",
                    "title": title,
                    "snippet": abstract[:500],
                    "url": url,
                    "domain": "crossref.org",
                    "category": "academic_paper",
                    "dataset_source": "crossref_live_api",
                    "relevance_score": 0.85,
                    "metadata": {
                        "source": "crossref",
                        "doi": doi,
                        "type": doc_type,
                        "published": pub_date,
                        "api_source": True,
                    },
                }

                # Generate embedding
                text_for_embedding = f"{title} {abstract}"
                embedding = self.sentence_model.encode(text_for_embedding)
                result_data["embedding"] = embedding.tolist()

                # Store in database
                self.db_handler.store_search_result(result_data)
                results.append(result_data)

            logger.info(f"Fetched {len(results)} results from CrossRef")
            return results

        except Exception as e:
            logger.error(f"Error fetching CrossRef results: {str(e)}")
            return []

    def _fetch_semantic_scholar_results(
        self, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """Fetch results from Semantic Scholar API"""
        try:
            self._rate_limit_wait("semantic_scholar")

            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,abstract,url,year,authors,citationCount",
            }

            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                timeout=15,
            )
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("data", []):
                title = item.get("title", "Untitled")
                abstract = item.get("abstract", "")
                url = item.get("url", "")
                year = item.get("year", "")
                citation_count = item.get("citationCount", 0)

                # Get authors
                authors = []
                for author in item.get("authors", []):
                    if "name" in author:
                        authors.append(author["name"])

                result_data = {
                    "query": query,
                    "language": "en",
                    "title": title,
                    "snippet": abstract[:500] if abstract else "",
                    "url": url,
                    "domain": "semanticscholar.org",
                    "category": "academic_paper",
                    "dataset_source": "semantic_scholar_live_api",
                    "relevance_score": min(
                        0.9, 0.7 + (citation_count / 1000)
                    ),  # Boost by citations
                    "metadata": {
                        "source": "semantic_scholar",
                        "year": year,
                        "authors": authors,
                        "citation_count": citation_count,
                        "api_source": True,
                    },
                }

                # Generate embedding
                text_for_embedding = f"{title} {abstract}" if abstract else title
                embedding = self.sentence_model.encode(text_for_embedding)
                result_data["embedding"] = embedding.tolist()

                # Store in database
                self.db_handler.store_search_result(result_data)
                results.append(result_data)

            logger.info(f"Fetched {len(results)} results from Semantic Scholar")
            return results

        except Exception as e:
            logger.error(f"Error fetching Semantic Scholar results: {str(e)}")
            return []

    def get_ambiguous_queries(self, language: str, limit: int) -> List[Dict[str, Any]]:
        """Get ambiguous queries - live APIs don't provide predefined queries"""
        return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for this source"""
        stats = self.db_handler.get_statistics()

        # Filter for live API sources
        live_sources = [
            k for k in stats.get("results_by_source", {}).keys() if "live_api" in k
        ]
        live_results = sum(
            stats.get("results_by_source", {}).get(source, 0) for source in live_sources
        )

        # Check which APIs are enabled
        enabled_apis = [name for name, config in self.apis.items() if config["enabled"]]

        return {
            "initialized": self.loaded,
            "total_results": live_results,
            "live_api_sources": live_sources,
            "enabled_apis": enabled_apis,
            "disabled_apis": [
                name for name in self.apis.keys() if name not in enabled_apis
            ],
            "supported_languages": ["en"],
            "api_configurations": {
                name: {"enabled": config["enabled"], "rate_limit": config["rate_limit"]}
                for name, config in self.apis.items()
            },
        }
