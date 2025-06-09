"""
TREC Dataset Source
Handles TREC Web Diversity Track datasets for real search evaluation data
"""

import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from ..core import DatasetSource
from ..database import DatabaseHandler

logger = logging.getLogger(__name__)


class TRECSource(DatasetSource):
    """
    TREC Web Diversity Track dataset source
    """

    def __init__(self, db_handler: DatabaseHandler, data_dir: str = "datasets/trec"):
        self.db_handler = db_handler
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.loaded = False

        # TREC download information
        self.download_info = {
            "base_url": "https://trec.nist.gov/data/",
            "files_needed": [
                "wt09-topics.xml",
                "wt10-topics.xml",
                "wt11-topics.xml",
                "wt12-topics.xml",
                "wt13-topics.xml",
            ],
            "qrels_files": [
                "wt09.qrels",
                "wt10.qrels",
                "wt11.qrels",
                "wt12.qrels",
                "wt13.qrels",
            ],
        }

        logger.info(f"TREC source initialized with data directory: {self.data_dir}")

    def load_data(self) -> bool:
        """Load TREC dataset files"""
        logger.info("Loading TREC Web Diversity Track data...")

        if not self._check_files_exist():
            logger.warning("TREC files not found. Please download them manually.")
            self._provide_download_instructions()
            return self._load_fallback_data()

        success_count = 0

        # Load topics files
        for topics_file in self.data_dir.glob("wt*-topics.xml"):
            try:
                if self._load_topics_file(topics_file):
                    success_count += 1
            except Exception as e:
                logger.error(f"Error loading {topics_file}: {str(e)}")

        # Load qrels files
        for qrels_file in self.data_dir.glob("*.qrels"):
            try:
                if self._load_qrels_file(qrels_file):
                    success_count += 1
            except Exception as e:
                logger.error(f"Error loading {qrels_file}: {str(e)}")

        self.loaded = success_count > 0
        logger.info(f"Loaded {success_count} TREC files")
        return self.loaded

    def _check_files_exist(self) -> bool:
        """Check if required TREC files exist"""
        topics_exist = any(self.data_dir.glob("wt*-topics.xml"))
        qrels_exist = any(self.data_dir.glob("*.qrels"))
        return topics_exist or qrels_exist

    def _provide_download_instructions(self):
        """Provide instructions for downloading TREC data"""
        instructions = f"""
        To use real TREC data, please download the following files:
        
        1. Visit: https://trec.nist.gov/data/webmain.html
        2. Download TREC Web Track topic files:
           - {', '.join(self.download_info['files_needed'])}
        
        3. Download relevance judgment files:
           - {', '.join(self.download_info['qrels_files'])}
        
        4. Place all files in: {self.data_dir}
        
        The system will use fallback sample data for now.
        """
        print(instructions)
        logger.info("TREC download instructions provided")

    def _load_fallback_data(self) -> bool:
        """Load sample TREC-style data when real files aren't available"""
        logger.info("Loading TREC fallback sample data...")

        # Sample TREC-style topics
        sample_topics = [
            {
                "number": "1",
                "query": "obama family tree",
                "description": "Find information about President Obama's family tree and genealogy.",
            },
            {
                "number": "2",
                "query": "french lick resort and casino",
                "description": "Find information about French Lick Resort and Casino in Indiana.",
            },
            {
                "number": "3",
                "query": "getting organized",
                "description": "Find tips and information about getting organized at home and work.",
            },
            {
                "number": "4",
                "query": "wedding budget calculator",
                "description": "Find tools and information for calculating wedding budgets.",
            },
            {
                "number": "5",
                "query": "map of the united states",
                "description": "Find maps showing the geography of the United States.",
            },
            {
                "number": "6",
                "query": "climate change",
                "description": "Find information about climate change causes and effects.",
            },
            {
                "number": "7",
                "query": "cooking tips",
                "description": "Find cooking tips and techniques for beginners.",
            },
            {
                "number": "8",
                "query": "job interview questions",
                "description": "Find common job interview questions and how to answer them.",
            },
        ]

        success_count = 0

        for topic in sample_topics:
            try:
                # Store as ambiguous query
                query_data = {
                    "query": topic["query"],
                    "language": "en",
                    "ambiguity_level": 0.8,
                    "entity_types": ["web_diversity"],
                    "dataset_source": "trec_sample",
                    "num_meanings": 5,
                }

                if self.db_handler.store_ambiguous_query(query_data):
                    success_count += 1

                # Create sample search results for this query
                self._create_sample_results(topic)

            except Exception as e:
                logger.error(f"Error storing sample topic {topic['number']}: {str(e)}")

        return success_count > 0

    def _create_sample_results(self, topic: Dict[str, str]):
        """Create sample search results for a topic"""
        query = topic["query"]
        description = topic["description"]

        # Generate diverse sample results
        sample_results = [
            {
                "title": f"Official Guide to {query.title()}",
                "snippet": f"{description} Comprehensive official information and resources.",
                "domain": "official.gov",
                "category": "government",
                "relevance": 0.95,
            },
            {
                "title": f"{query.title()} - Wikipedia",
                "snippet": f"Wikipedia article about {query} with detailed information and references.",
                "domain": "wikipedia.org",
                "category": "reference",
                "relevance": 0.90,
            },
            {
                "title": f"How to Guide: {query.title()}",
                "snippet": f"Step-by-step guide and tips for {query}. Practical advice and examples.",
                "domain": "wikihow.com",
                "category": "tutorial",
                "relevance": 0.85,
            },
            {
                "title": f"{query.title()} News and Updates",
                "snippet": f"Latest news and recent developments related to {query}.",
                "domain": "news.com",
                "category": "news",
                "relevance": 0.80,
            },
            {
                "title": f"{query.title()} Forum Discussion",
                "snippet": f"Community discussion and user experiences with {query}.",
                "domain": "reddit.com",
                "category": "forum",
                "relevance": 0.70,
            },
        ]

        for i, result_template in enumerate(sample_results):
            result_data = {
                "query": query,
                "language": "en",
                "title": result_template["title"],
                "snippet": result_template["snippet"],
                "url": f"https://www.{result_template['domain']}/{query.replace(' ', '-')}",
                "domain": result_template["domain"],
                "category": result_template["category"],
                "dataset_source": "trec_sample",
                "relevance_score": result_template["relevance"],
                "metadata": {
                    "topic_number": topic["number"],
                    "topic_description": description,
                    "result_rank": i + 1,
                },
            }

            # Generate embedding
            text_for_embedding = f"{result_data['title']} {result_data['snippet']}"
            embedding = self.sentence_model.encode(text_for_embedding)
            result_data["embedding"] = embedding.tolist()

            self.db_handler.store_search_result(result_data)

    def _load_topics_file(self, topics_file: Path) -> bool:
        """Load TREC topics XML file"""
        try:
            tree = ET.parse(topics_file)
            root = tree.getroot()

            for topic in root.findall(".//topic"):
                number = topic.get("number")
                query_elem = topic.find("query")
                description_elem = topic.find("description")

                if query_elem is not None:
                    query_text = query_elem.text.strip()
                    description = (
                        description_elem.text.strip()
                        if description_elem is not None
                        else ""
                    )

                    # Store as ambiguous query
                    query_data = {
                        "query": query_text,
                        "language": "en",
                        "ambiguity_level": 0.8,
                        "entity_types": ["web_diversity"],
                        "dataset_source": f"trec_{topics_file.stem}",
                        "num_meanings": 5,
                    }

                    self.db_handler.store_ambiguous_query(query_data)

                    # Create placeholder search result
                    result_data = {
                        "query": query_text,
                        "language": "en",
                        "title": f"TREC Topic {number}: {query_text}",
                        "snippet": description,
                        "url": f"https://trec.nist.gov/topics/{number}",
                        "domain": "trec.nist.gov",
                        "category": "research_topic",
                        "dataset_source": f"trec_{topics_file.stem}",
                        "relevance_score": 0.9,
                        "metadata": {
                            "topic_number": number,
                            "description": description,
                            "track": topics_file.stem,
                        },
                    }

                    # Generate embedding
                    text_for_embedding = f"{query_text} {description}"
                    embedding = self.sentence_model.encode(text_for_embedding)
                    result_data["embedding"] = embedding.tolist()

                    self.db_handler.store_search_result(result_data)

            logger.info(f"Loaded TREC topics from {topics_file}")
            return True

        except Exception as e:
            logger.error(f"Error parsing TREC topics {topics_file}: {str(e)}")
            return False

    def _load_qrels_file(self, qrels_file: Path) -> bool:
        """Load TREC qrels (relevance judgments) file"""
        try:
            with open(qrels_file, "r") as f:
                count = 0
                for line in f:
                    parts = (
                        line.strip().split("\t")
                        if "\t" in line
                        else line.strip().split()
                    )
                    if len(parts) >= 4:
                        topic_id, iteration, doc_id, relevance = parts[:4]

                        # Store relevance mapping
                        self.db_handler.store_relevance_mapping(
                            f"trec_topic_{topic_id}",
                            f"trec_doc_{doc_id}",
                            float(relevance),
                            f"trec_qrels_{qrels_file.stem}",
                        )

                        # Create document result if high relevance
                        if float(relevance) >= 1:
                            result_data = {
                                "query": f"topic_{topic_id}",
                                "language": "en",
                                "title": f"TREC Document {doc_id}",
                                "snippet": f"Relevant document for TREC topic {topic_id} with relevance score {relevance}",
                                "url": f"https://trec.docs/{doc_id}",
                                "domain": "trec.nist.gov",
                                "category": "web_document",
                                "dataset_source": f"trec_qrels_{qrels_file.stem}",
                                "relevance_score": float(relevance)
                                / 3.0,  # Normalize to 0-1
                                "metadata": {
                                    "topic_id": topic_id,
                                    "doc_id": doc_id,
                                    "relevance": relevance,
                                    "iteration": iteration,
                                },
                            }

                            # Generate embedding
                            text_for_embedding = (
                                f"{result_data['title']} {result_data['snippet']}"
                            )
                            embedding = self.sentence_model.encode(text_for_embedding)
                            result_data["embedding"] = embedding.tolist()

                            self.db_handler.store_search_result(result_data)

                        count += 1

            logger.info(f"Loaded {count} relevance judgments from {qrels_file}")
            return True

        except Exception as e:
            logger.error(f"Error parsing TREC qrels {qrels_file}: {str(e)}")
            return False

    def get_results(
        self, query: str, language: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get search results for query"""
        if language != "en":
            return []  # TREC is English only

        # Get results from different TREC sources
        sources = [
            "trec_sample",
            "trec_wt09-topics",
            "trec_wt10-topics",
            "trec_wt11-topics",
            "trec_wt12-topics",
            "trec_wt13-topics",
        ]

        all_results = []
        for source in sources:
            results = self.db_handler.get_search_results(query, language, limit, source)
            all_results.extend(results)

        # Sort by relevance and limit
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return all_results[:limit]

    def get_ambiguous_queries(self, language: str, limit: int) -> List[Dict[str, Any]]:
        """Get ambiguous queries"""
        if language != "en":
            return []

        sources = [
            "trec_sample",
            "trec_wt09-topics",
            "trec_wt10-topics",
            "trec_wt11-topics",
            "trec_wt12-topics",
            "trec_wt13-topics",
        ]

        all_queries = []
        for source in sources:
            queries = self.db_handler.get_ambiguous_queries(language, limit, source)
            all_queries.extend(queries)

        # Remove duplicates and sort
        seen = set()
        unique_queries = []
        for query in all_queries:
            key = query["query"]
            if key not in seen:
                seen.add(key)
                unique_queries.append(query)

        unique_queries.sort(key=lambda x: x.get("ambiguity_level", 0), reverse=True)
        return unique_queries[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for this source"""
        stats = self.db_handler.get_statistics()

        # Filter for TREC sources
        trec_sources = [
            k
            for k in stats.get("results_by_source", {}).keys()
            if k.startswith("trec_")
        ]
        trec_results = sum(
            stats.get("results_by_source", {}).get(source, 0) for source in trec_sources
        )

        return {
            "initialized": self.loaded,
            "total_results": trec_results,
            "trec_sources": trec_sources,
            "data_directory": str(self.data_dir),
            "files_available": list(self.data_dir.glob("*")),
            "supported_languages": ["en"],
        }
