"""
MIRACL Dataset Source - Simplified Direct Download
Downloads MIRACL data directly from HuggingFace without datasets library dependency
"""

import logging
import json
import requests
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
from datetime import datetime
import time

from ..core import DatasetSource
from ..database import DatabaseHandler

logger = logging.getLogger(__name__)


class MIRACLSource(DatasetSource):
    """
    MIRACL dataset source with direct file downloads from HuggingFace
    """

    def __init__(self, db_handler: Optional[DatabaseHandler], data_dir: str):
        self.db_handler = db_handler
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.initialized = False
        self.available_languages = ["ar", "en"]  # Focus on these for the project

        # MIRACL file URLs (corrected based on actual HuggingFace structure)
        self.base_urls = {
            "ar": {
                "topics_dev": "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-ar/topics/topics.miracl-v1.0-ar-dev.tsv",
                "qrels_dev": "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-ar/qrels/qrels.miracl-v1.0-ar-dev.tsv",
                "topics_train": "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-ar/topics/topics.miracl-v1.0-ar-train.tsv",
                "qrels_train": "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-ar/qrels/qrels.miracl-v1.0-ar-train.tsv",
            },
            "en": {
                "topics_dev": "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-en/topics/topics.miracl-v1.0-en-dev.tsv",
                "qrels_dev": "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-dev.tsv",
                "topics_train": "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-en/topics/topics.miracl-v1.0-en-train.tsv",
                "qrels_train": "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-train.tsv",
            },
        }

        logger.info(f"MIRACL source initialized with data directory: {self.data_dir}")

    def load_data(self) -> bool:
        """Load MIRACL data by downloading and processing files"""
        try:
            success_count = 0

            for lang in self.available_languages:
                if self._download_and_process_language(lang):
                    success_count += 1

            self.initialized = success_count > 0

            if self.initialized:
                logger.info(
                    f"MIRACL data loaded successfully for {success_count} languages"
                )
            else:
                logger.warning("Failed to load any MIRACL datasets")

            return self.initialized

        except Exception as e:
            logger.error(f"Error loading MIRACL data: {str(e)}")
            return False

    def _download_and_process_language(self, language: str) -> bool:
        """Download and process MIRACL data for a specific language"""
        try:
            logger.info(f"Processing MIRACL dataset for language: {language}")

            lang_dir = self.data_dir / language
            lang_dir.mkdir(exist_ok=True)

            # Download topics and qrels files
            topics_dev = self._download_file(
                self.base_urls[language]["topics_dev"], lang_dir / "topics_dev.tsv"
            )

            qrels_dev = self._download_file(
                self.base_urls[language]["qrels_dev"], lang_dir / "qrels_dev.tsv"
            )

            if not (topics_dev and qrels_dev):
                logger.warning(f"Failed to download dev files for {language}")
                return False

            # Try to download train files (may not exist for all languages)
            topics_train = self._download_file(
                self.base_urls[language]["topics_train"],
                lang_dir / "topics_train.tsv",
                required=False,
            )

            qrels_train = self._download_file(
                self.base_urls[language]["qrels_train"],
                lang_dir / "qrels_train.tsv",
                required=False,
            )

            # Process the downloaded files
            if self.db_handler:
                self._process_language_files(language, lang_dir)

            return True

        except Exception as e:
            logger.error(f"Error processing {language}: {str(e)}")
            return False

    def _download_file(self, url: str, file_path: Path, required: bool = True) -> bool:
        """Download a file from URL"""
        try:
            if file_path.exists():
                logger.info(f"File already exists: {file_path.name}")
                return True

            logger.info(f"Downloading: {url}")

            response = requests.get(url, stream=True, timeout=30)

            if response.status_code == 404:
                if required:
                    logger.error(f"Required file not found: {url}")
                    return False
                else:
                    logger.info(f"Optional file not found: {url}")
                    return True

            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded: {file_path.name}")
            return True

        except Exception as e:
            if required:
                logger.error(f"Error downloading {url}: {str(e)}")
                return False
            else:
                logger.warning(f"Optional download failed {url}: {str(e)}")
                return True

    def _process_language_files(self, language: str, lang_dir: Path):
        """Process downloaded files and store in database"""
        if not self.db_handler:
            return

        logger.info(f"Processing files for {language}")

        # Process dev split
        topics_dev = lang_dir / "topics_dev.tsv"
        qrels_dev = lang_dir / "qrels_dev.tsv"

        if topics_dev.exists() and qrels_dev.exists():
            self._process_split(language, "dev", topics_dev, qrels_dev)

        # Process train split if available
        topics_train = lang_dir / "topics_train.tsv"
        qrels_train = lang_dir / "qrels_train.tsv"

        if topics_train.exists() and qrels_train.exists():
            self._process_split(language, "train", topics_train, qrels_train)

    def _process_split(
        self, language: str, split_name: str, topics_file: Path, qrels_file: Path
    ):
        """Process a specific split (dev/train)"""
        try:
            # Load topics (queries)
            topics = self._load_topics(topics_file)

            # Load qrels (relevance judgments)
            qrels = self._load_qrels(qrels_file)

            logger.info(f"Processing {len(topics)} topics for {language} {split_name}")

            processed_count = 0

            for query_id, query_text in topics.items():
                try:
                    # Store query as ambiguous
                    query_data = {
                        "id": self._generate_id(f"miracl_{language}_{query_id}"),
                        "query": query_text,
                        "language": language,
                        "ambiguity_level": self._estimate_ambiguity(query_text),
                        "entity_types": self._extract_entity_types(query_text),
                        "dataset_source": f"miracl_{split_name}",
                        "num_meanings": len(qrels.get(query_id, {})),
                    }

                    self.db_handler.store_ambiguous_query(query_data)

                    # Process relevance judgments for this query
                    if query_id in qrels:
                        for doc_id, relevance in qrels[query_id].items():
                            # Create a synthetic search result
                            result_data = {
                                "id": self._generate_id(f"miracl_{doc_id}_{query_id}"),
                                "query": query_text,
                                "language": language,
                                "title": f"Document {doc_id}",
                                "snippet": f"Relevant document for query: {query_text}",
                                "url": f"https://miracl.ai/doc/{doc_id}",
                                "domain": "miracl.ai",
                                "category": self._categorize_query(query_text),
                                "dataset_source": f"miracl_{split_name}",
                                "relevance_score": float(relevance),
                                "metadata": {
                                    "doc_id": doc_id,
                                    "query_id": query_id,
                                    "split": split_name,
                                    "original_relevance": relevance,
                                },
                            }

                            self.db_handler.store_search_result(result_data)

                            # Store relevance mapping
                            self.db_handler.store_relevance_mapping(
                                query_data["id"],
                                result_data["id"],
                                float(relevance),
                                f"miracl_{split_name}",
                            )

                    processed_count += 1

                    if processed_count % 50 == 0:
                        logger.info(
                            f"Processed {processed_count} queries for {language} {split_name}"
                        )

                except Exception as e:
                    logger.warning(f"Error processing query {query_id}: {str(e)}")
                    continue

            logger.info(
                f"Completed processing {processed_count} queries for {language} {split_name}"
            )

        except Exception as e:
            logger.error(f"Error processing split {split_name}: {str(e)}")

    def _load_topics(self, topics_file: Path) -> Dict[str, str]:
        """Load topics from TSV file"""
        topics = {}

        try:
            with open(topics_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "\t" in line:
                        parts = line.split("\t", 1)
                        if len(parts) == 2:
                            query_id, query_text = parts
                            topics[query_id] = query_text

            logger.info(f"Loaded {len(topics)} topics from {topics_file.name}")
            return topics

        except Exception as e:
            logger.error(f"Error loading topics from {topics_file}: {str(e)}")
            return {}

    def _load_qrels(self, qrels_file: Path) -> Dict[str, Dict[str, int]]:
        """Load qrels from TSV file"""
        qrels = {}

        try:
            with open(qrels_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split("\t")
                        if len(parts) >= 4:
                            query_id, _, doc_id, relevance = parts[:4]

                            if query_id not in qrels:
                                qrels[query_id] = {}

                            qrels[query_id][doc_id] = int(relevance)

            logger.info(f"Loaded qrels for {len(qrels)} queries from {qrels_file.name}")
            return qrels

        except Exception as e:
            logger.error(f"Error loading qrels from {qrels_file}: {str(e)}")
            return {}

    def get_results(
        self, query: str, language: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get search results for a query"""
        if not self.initialized or not self.db_handler:
            return []

        results = self.db_handler.get_search_results(
            query, language, limit, source=None
        )

        # Filter for MIRACL sources only
        miracl_results = [
            r for r in results if r["dataset_source"].startswith("miracl_")
        ]

        return miracl_results

    def get_ambiguous_queries(self, language: str, limit: int) -> List[Dict[str, Any]]:
        """Get ambiguous queries from MIRACL"""
        if not self.initialized or not self.db_handler:
            return []

        queries = self.db_handler.get_ambiguous_queries(language, limit, source=None)

        # Filter for MIRACL sources only
        miracl_queries = [
            q for q in queries if q["dataset_source"].startswith("miracl_")
        ]

        return miracl_queries

    def _estimate_ambiguity(self, query: str) -> float:
        """Estimate ambiguity level of a query"""
        query_lower = query.lower()

        # Base ambiguity score
        ambiguity_score = 0.5

        # Short queries tend to be more ambiguous
        word_count = len(query.split())
        if word_count == 1:
            ambiguity_score += 0.3
        elif word_count == 2:
            ambiguity_score += 0.2

        # Common ambiguous terms
        ambiguous_terms = {
            "en": ["apple", "python", "mercury", "mars", "amazon", "oracle", "jaguar"],
            "ar": ["عين", "بنك", "ورد", "سلم", "نور", "قلم", "جوز"],
        }

        # Detect language from query characters
        if any("\u0600" <= char <= "\u06ff" for char in query):
            query_lang = "ar"
        else:
            query_lang = "en"

        if query_lang in ambiguous_terms:
            for term in ambiguous_terms[query_lang]:
                if term in query_lower:
                    ambiguity_score += 0.2
                    break

        return min(1.0, ambiguity_score)

    def _extract_entity_types(self, query: str) -> List[str]:
        """Extract potential entity types from query"""
        query_lower = query.lower()
        entity_types = []

        # Simple heuristics for entity detection
        if any(
            indicator in query_lower
            for indicator in ["person", "people", "man", "woman"]
        ):
            entity_types.append("person")

        if any(
            indicator in query_lower
            for indicator in ["company", "corporation", "business"]
        ):
            entity_types.append("organization")

        if any(
            indicator in query_lower
            for indicator in ["city", "country", "place", "location"]
        ):
            entity_types.append("location")

        if any(
            indicator in query_lower for indicator in ["book", "movie", "song", "film"]
        ):
            entity_types.append("creative_work")

        return entity_types if entity_types else ["general"]

    def _categorize_query(self, query: str) -> str:
        """Categorize query based on content"""
        query_lower = query.lower()

        if any(keyword in query_lower for keyword in ["person", "people", "biography"]):
            return "person"
        elif any(
            keyword in query_lower for keyword in ["company", "business", "corporation"]
        ):
            return "organization"
        elif any(keyword in query_lower for keyword in ["city", "country", "location"]):
            return "location"
        elif any(
            keyword in query_lower for keyword in ["technology", "computer", "software"]
        ):
            return "technology"
        elif any(
            keyword in query_lower for keyword in ["science", "research", "study"]
        ):
            return "science"
        else:
            return "general"

    def _generate_id(self, content: str) -> str:
        """Generate consistent ID from content"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]

    def get_statistics(self) -> Dict[str, Any]:
        """Get MIRACL dataset statistics"""
        stats = {
            "initialized": self.initialized,
            "total_results": 0,
            "total_queries": 0,
            "languages": self.available_languages,
            "splits": {},
        }

        if self.initialized and self.db_handler:
            db_stats = self.db_handler.get_statistics()

            # Count MIRACL-specific data
            for source, count in db_stats.get("results_by_source", {}).items():
                if source.startswith("miracl_"):
                    stats["total_results"] += count
                    stats["splits"][source] = count

            # Get query count from database
            for lang in self.available_languages:
                queries = self.db_handler.get_ambiguous_queries(lang, 1000)
                miracl_queries = [
                    q for q in queries if q["dataset_source"].startswith("miracl_")
                ]
                stats["total_queries"] += len(miracl_queries)

        return stats

    def download_sample_data(self) -> bool:
        """
        Download a small sample of MIRACL data for testing
        """
        try:
            logger.info("Downloading MIRACL sample data...")

            # Create sample data manually since we can't use HF datasets
            sample_data = {
                "ar": [
                    {
                        "query_id": "ar_sample_1",
                        "query": "محمد صلاح",
                        "relevance_docs": {"doc1": 1, "doc2": 1, "doc3": 0},
                    },
                    {
                        "query_id": "ar_sample_2",
                        "query": "الهرم الأكبر",
                        "relevance_docs": {"doc4": 1, "doc5": 1, "doc6": 0},
                    },
                    {
                        "query_id": "ar_sample_3",
                        "query": "عين",
                        "relevance_docs": {"doc7": 1, "doc8": 1, "doc9": 0, "doc10": 0},
                    },
                ],
                "en": [
                    {
                        "query_id": "en_sample_1",
                        "query": "python programming",
                        "relevance_docs": {"doc11": 1, "doc12": 1, "doc13": 0},
                    },
                    {
                        "query_id": "en_sample_2",
                        "query": "apple",
                        "relevance_docs": {
                            "doc14": 1,
                            "doc15": 1,
                            "doc16": 0,
                            "doc17": 0,
                        },
                    },
                ],
            }

            # Save sample data
            sample_file = self.data_dir / "miracl_sample.json"
            with open(sample_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Sample data created at {sample_file}")
            return True

        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")
            return False

    def load_from_sample(self) -> bool:
        """Load data from sample file"""
        sample_file = self.data_dir / "miracl_sample.json"

        if not sample_file.exists():
            # Create sample data if it doesn't exist
            if not self.download_sample_data():
                return False

        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                sample_data = json.load(f)

            if not self.db_handler:
                logger.warning("No database handler available")
                return False

            total_processed = 0

            for language, queries in sample_data.items():
                for query_item in queries:
                    # Store query
                    query_data = {
                        "id": self._generate_id(
                            f"miracl_{language}_{query_item['query_id']}"
                        ),
                        "query": query_item["query"],
                        "language": language,
                        "ambiguity_level": self._estimate_ambiguity(
                            query_item["query"]
                        ),
                        "entity_types": self._extract_entity_types(query_item["query"]),
                        "dataset_source": "miracl_sample",
                        "num_meanings": len(query_item["relevance_docs"]),
                    }

                    self.db_handler.store_ambiguous_query(query_data)

                    # Store documents
                    for doc_id, relevance in query_item["relevance_docs"].items():
                        result_data = {
                            "id": self._generate_id(
                                f"miracl_{doc_id}_{query_item['query_id']}"
                            ),
                            "query": query_item["query"],
                            "language": language,
                            "title": f"Document {doc_id} for {query_item['query']}",
                            "snippet": f"This is a {'relevant' if relevance else 'non-relevant'} document for the query '{query_item['query']}'.",
                            "url": f"https://miracl.ai/doc/{doc_id}",
                            "domain": "miracl.ai",
                            "category": self._categorize_query(query_item["query"]),
                            "dataset_source": "miracl_sample",
                            "relevance_score": float(relevance),
                            "metadata": {
                                "doc_id": doc_id,
                                "query_id": query_item["query_id"],
                                "is_sample": True,
                            },
                        }

                        self.db_handler.store_search_result(result_data)

                        # Store relevance mapping
                        self.db_handler.store_relevance_mapping(
                            query_data["id"],
                            result_data["id"],
                            float(relevance),
                            "miracl_sample",
                        )

                    total_processed += 1

            self.initialized = True
            logger.info(
                f"Sample data loaded successfully: {total_processed} queries processed"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading sample data: {str(e)}")
            return False
