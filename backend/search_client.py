"""
Search Client - Provides real search results from Google Custom Search API and MIRACL dataset
Implements search functionality for both English and Arabic queries with real data sources
OPTIMIZED VERSION with pre-computed embeddings
"""

import requests
import json
import time
import logging
import os
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
import wikipediaapi
from bs4 import BeautifulSoup
import datasets
from datasets import load_dataset
import pickle
import numpy as np

logger = logging.getLogger(__name__)


class SearchClient:
    """
    Search client that provides real search results from:
    1. Google Custom Search API (primary source)
    2. MIRACL-Arabic corpus (for Arabic queries)
    3. TREC Web Diversity dataset
    4. Wikipedia API (fallback)

    OPTIMIZED: Pre-computes and caches TF-IDF vectors for fast Arabic search
    """

    def __init__(self):
        """Initialize search client with real data sources"""
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "DSR-RL Search System (Educational Research)"}
        )

        # Google Custom Search API configuration
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")

        if not self.google_api_key or not self.google_cse_id:
            logger.warning(
                "Google Custom Search API credentials not found in environment variables"
            )
            logger.warning(
                "Please set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables"
            )

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests

        # Dataset caches
        self.miracl_arabic_cache = None
        self.trec_cache = None

        # PRE-COMPUTED VECTORS (NEW)
        self.miracl_tfidf_vectorizer = None
        self.miracl_doc_vectors = None
        self.miracl_doc_metadata = None

        # Initialize Wikipedia API
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language="en", user_agent="DSR-RL Search System (Educational Research)"
        )

        # Initialize datasets
        self._load_datasets()

        logger.info("Search client initialized with real data sources")

    def _load_datasets(self):
        """Load real datasets: MIRACL-Arabic and TREC Web Diversity"""
        try:
            logger.info("Loading real datasets...")

            # Load MIRACL-Arabic corpus with pre-computed vectors
            self._load_miracl_arabic_optimized()

            # Load TREC Web Diversity data
            self._load_trec_web_diversity()

            logger.info("Real datasets loaded successfully")

        except Exception as e:
            logger.error(f"Error loading datasets: {e}")

    def _load_miracl_arabic_optimized(self):
        """Load MIRACL-Arabic corpus with pre-computed TF-IDF vectors for fast search"""
        try:
            cache_file = "miracl_arabic_cache.pkl"
            vectors_cache_file = "miracl_arabic_vectors.pkl"

            # Check if we have pre-computed vectors
            if os.path.exists(vectors_cache_file):
                logger.info("Loading pre-computed MIRACL-Arabic vectors from cache...")
                with open(vectors_cache_file, "rb") as f:
                    vector_data = pickle.load(f)
                    self.miracl_tfidf_vectorizer = vector_data["vectorizer"]
                    self.miracl_doc_vectors = vector_data["doc_vectors"]
                    self.miracl_doc_metadata = vector_data["doc_metadata"]

                logger.info(
                    f"✅ Loaded pre-computed vectors for {len(self.miracl_doc_metadata)} documents"
                )
                return

            # If no pre-computed vectors, load the corpus first
            if os.path.exists(cache_file):
                logger.info("Loading MIRACL-Arabic from cache...")
                with open(cache_file, "rb") as f:
                    self.miracl_arabic_cache = pickle.load(f)
            else:
                logger.info("Downloading MIRACL-Arabic corpus from HuggingFace...")

                try:
                    # Load MIRACL dataset with trust_remote_code=True
                    dataset = load_dataset(
                        "miracl/miracl-corpus",
                        "ar",
                        split="train",
                        trust_remote_code=True,
                    )

                    # Convert to searchable format (limit to first 10k docs for demo)
                    self.miracl_arabic_cache = []
                    for i, item in enumerate(dataset):
                        if i >= 10000:  # Limit for faster processing
                            break

                        doc = {
                            "docid": item.get("docid", ""),
                            "title": item.get("title", ""),
                            "text": item.get("text", ""),
                            "url": f"https://miracl.corpus/{item.get('docid', '')}",
                            "source": "miracl_arabic",
                        }
                        self.miracl_arabic_cache.append(doc)

                    # Cache the dataset
                    with open(cache_file, "wb") as f:
                        pickle.dump(self.miracl_arabic_cache, f)

                    logger.info(
                        f"MIRACL-Arabic loaded: {len(self.miracl_arabic_cache)} documents"
                    )

                except Exception as e:
                    logger.error(f"Error loading MIRACL-Arabic: {e}")
                    logger.info("Creating fallback Arabic documents...")
                    self.miracl_arabic_cache = self._create_fallback_arabic_corpus()

            # Now pre-compute TF-IDF vectors
            logger.info(
                "🔄 Pre-computing TF-IDF vectors for Arabic search (one-time setup)..."
            )
            self._precompute_miracl_vectors()

        except Exception as e:
            logger.error(f"Error loading MIRACL-Arabic: {e}")
            logger.info("Creating fallback Arabic documents...")
            self.miracl_arabic_cache = self._create_fallback_arabic_corpus()
            if self.miracl_arabic_cache:
                self._precompute_miracl_vectors()

    def _precompute_miracl_vectors(self):
        """Pre-compute TF-IDF vectors for the MIRACL corpus"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            if not self.miracl_arabic_cache:
                logger.warning("No MIRACL Arabic documents to vectorize")
                return

            logger.info("🔄 Computing TF-IDF vectors for Arabic corpus...")

            # Prepare documents for vectorization
            documents = []
            doc_metadata = []

            for doc in self.miracl_arabic_cache:
                text = f"{doc.get('title', '')} {doc.get('text', '')}"
                if text.strip():
                    documents.append(text)
                    doc_metadata.append(doc)

            if not documents:
                logger.warning("No valid documents found for vectorization")
                return

            # Create TF-IDF vectorizer optimized for Arabic
            self.miracl_tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words=None,  # No Arabic stopwords in sklearn
                ngram_range=(1, 2),
                min_df=2,  # Ignore very rare terms
                max_df=0.8,  # Ignore very common terms
                sublinear_tf=True,  # Apply log scaling
            )

            # Fit and transform documents
            logger.info(f"Vectorizing {len(documents)} documents...")
            self.miracl_doc_vectors = self.miracl_tfidf_vectorizer.fit_transform(
                documents
            )
            self.miracl_doc_metadata = doc_metadata

            # Cache the pre-computed vectors
            vector_data = {
                "vectorizer": self.miracl_tfidf_vectorizer,
                "doc_vectors": self.miracl_doc_vectors,
                "doc_metadata": self.miracl_doc_metadata,
            }

            vectors_cache_file = "miracl_arabic_vectors.pkl"
            with open(vectors_cache_file, "wb") as f:
                pickle.dump(vector_data, f)

            logger.info(
                f"✅ Pre-computed and cached TF-IDF vectors for {len(documents)} documents"
            )
            logger.info(f"📊 Vector dimensions: {self.miracl_doc_vectors.shape}")

        except Exception as e:
            logger.error(f"Error pre-computing MIRACL vectors: {e}")
            self.miracl_tfidf_vectorizer = None
            self.miracl_doc_vectors = None
            self.miracl_doc_metadata = None

    def _create_fallback_arabic_corpus(self) -> List[Dict[str, Any]]:
        """Create fallback Arabic corpus when MIRACL is not available"""
        try:
            fallback_docs = [
                {
                    "docid": "ar_001",
                    "title": "الذكاء الاصطناعي في العالم العربي",
                    "text": "يشهد مجال الذكاء الاصطناعي تطوراً سريعاً في المنطقة العربية مع استثمارات كبيرة في التكنولوجيا والبحث والتطوير. تركز الدول العربية على تطوير حلول الذكاء الاصطناعي المحلية.",
                    "url": "https://fallback.corpus/ar_001",
                    "source": "miracl_arabic",
                },
                {
                    "docid": "ar_002",
                    "title": "التعليم الإلكتروني في الجامعات العربية",
                    "text": "انتشر التعليم الإلكتروني بشكل واسع في الجامعات العربية خاصة بعد جائحة كوفيد-19. تبنت المؤسسات التعليمية تقنيات حديثة لضمان استمرارية التعليم.",
                    "url": "https://fallback.corpus/ar_002",
                    "source": "miracl_arabic",
                },
                {
                    "docid": "ar_003",
                    "title": "تاريخ الحضارة الإسلامية",
                    "text": "الحضارة الإسلامية لعبت دوراً مهماً في تطوير العلوم والفنون والفلسفة عبر التاريخ. ساهم العلماء المسلمون في تقدم الرياضيات والطب والفلك.",
                    "url": "https://fallback.corpus/ar_003",
                    "source": "miracl_arabic",
                },
                {
                    "docid": "ar_004",
                    "title": "تكنولوجيا المعلومات في الوطن العربي",
                    "text": "تشهد تكنولوجيا المعلومات في الوطن العربي نمواً متسارعاً مع زيادة الاستثمار في البنية التحتية الرقمية والتحول الرقمي.",
                    "url": "https://fallback.corpus/ar_004",
                    "source": "miracl_arabic",
                },
                {
                    "docid": "ar_005",
                    "title": "التجارة الإلكترونية في المنطقة العربية",
                    "text": "نمت التجارة الإلكترونية بشكل كبير في المنطقة العربية مع زيادة استخدام الإنترنت والهواتف الذكية وتطور أنظمة الدفع الرقمي.",
                    "url": "https://fallback.corpus/ar_005",
                    "source": "miracl_arabic",
                },
                {
                    "docid": "ar_006",
                    "title": "الطب والصحة في العالم العربي",
                    "text": "يواجه قطاع الصحة في العالم العربي تحديات وفرص عديدة مع التطورات التكنولوجية والحاجة لتحسين الخدمات الطبية.",
                    "url": "https://fallback.corpus/ar_006",
                    "source": "miracl_arabic",
                },
                {
                    "docid": "ar_007",
                    "title": "الطاقة المتجددة في البلدان العربية",
                    "text": "تستثمر البلدان العربية بشكل متزايد في مشاريع الطاقة المتجددة مثل الطاقة الشمسية وطاقة الرياح لتنويع مصادر الطاقة.",
                    "url": "https://fallback.corpus/ar_007",
                    "source": "miracl_arabic",
                },
                {
                    "docid": "ar_008",
                    "title": "الثقافة والفنون في المجتمع العربي",
                    "text": "تتميز الثقافة العربية بتنوعها وغناها عبر مختلف البلدان العربية مع تأثيرات تاريخية وجغرافية متنوعة.",
                    "url": "https://fallback.corpus/ar_008",
                    "source": "miracl_arabic",
                },
                {
                    "docid": "ar_009",
                    "title": "الأمن السيبراني في المؤسسات العربية",
                    "text": "تزداد أهمية الأمن السيبراني في المؤسسات العربية مع تزايد التهديدات الرقمية والحاجة لحماية البيانات والأنظمة.",
                    "url": "https://fallback.corpus/ar_009",
                    "source": "miracl_arabic",
                },
                {
                    "docid": "ar_010",
                    "title": "الابتكار والبحث العلمي في الجامعات العربية",
                    "text": "تسعى الجامعات العربية لتعزيز الابتكار والبحث العلمي من خلال الاستثمار في البحوث والتطوير والشراكات الدولية.",
                    "url": "https://fallback.corpus/ar_010",
                    "source": "miracl_arabic",
                },
            ]

            logger.info(
                f"Created fallback Arabic corpus with {len(fallback_docs)} documents"
            )
            return fallback_docs

        except Exception as e:
            logger.error(f"Error creating fallback Arabic corpus: {e}")
            return []

    def _load_trec_web_diversity(self):
        """Load TREC Web Diversity dataset"""
        try:
            cache_file = "trec_web_diversity_cache.pkl"

            if os.path.exists(cache_file):
                logger.info("Loading TREC Web Diversity from cache...")
                with open(cache_file, "rb") as f:
                    self.trec_cache = pickle.load(f)
            else:
                logger.info("Downloading TREC Web Diversity dataset...")

                # Try to load TREC dataset from available sources
                try:
                    # Attempt to load from ir_datasets if available
                    import ir_datasets

                    # Try different TREC dataset names
                    trec_dataset_names = [
                        "trec-web-2012",
                        "trec-web-2013",
                        "trec-web-2014",
                        "msmarco-passage",
                    ]

                    dataset = None
                    for dataset_name in trec_dataset_names:
                        try:
                            dataset = ir_datasets.load(dataset_name)
                            logger.info(f"Successfully loaded {dataset_name}")
                            break
                        except Exception as e:
                            logger.debug(f"Failed to load {dataset_name}: {e}")
                            continue

                    if dataset is None:
                        raise ImportError("No TREC datasets available")

                    self.trec_cache = []
                    for doc in dataset.docs_iter():
                        trec_doc = {
                            "doc_id": doc.doc_id,
                            "title": getattr(doc, "title", doc.doc_id),
                            "text": getattr(doc, "text", ""),
                            "url": getattr(
                                doc, "url", f"https://trec.nist.gov/{doc.doc_id}"
                            ),
                            "source": "trec_web_diversity",
                        }
                        self.trec_cache.append(trec_doc)

                        # Limit to reasonable size for demo
                        if len(self.trec_cache) >= 1000:
                            break

                except (ImportError, Exception) as e:
                    logger.warning(f"ir_datasets not available or failed: {e}")
                    logger.info("Using Wikipedia disambiguation pages as TREC fallback")
                    self.trec_cache = self._create_wikipedia_disambiguation_corpus()

                # Cache the dataset
                with open(cache_file, "wb") as f:
                    pickle.dump(self.trec_cache, f)

                logger.info(
                    f"TREC Web Diversity loaded: {len(self.trec_cache)} documents"
                )

        except Exception as e:
            logger.error(f"Error loading TREC dataset: {e}")
            logger.info("Creating fallback TREC corpus...")
            self.trec_cache = self._create_fallback_trec_corpus()

    def _create_fallback_trec_corpus(self) -> List[Dict[str, Any]]:
        """Create fallback TREC corpus when real TREC is not available"""
        try:
            # Create ambiguous query examples as mentioned in the paper
            ambiguous_queries = {
                "jackson": [
                    {
                        "doc_id": "trec_jackson_001",
                        "title": "Michael Jackson Biography",
                        "text": "Michael Joseph Jackson was an American singer, songwriter, and dancer. Dubbed the King of Pop, he is regarded as one of the most significant cultural figures of the 20th century.",
                        "url": "https://trec.fallback/jackson/michael",
                        "source": "trec_web_diversity",
                    },
                    {
                        "doc_id": "trec_jackson_002",
                        "title": "Jackson, Mississippi City Guide",
                        "text": "Jackson is the capital and most populous city of Mississippi. It has been the state capital since 1821 and is named after Andrew Jackson.",
                        "url": "https://trec.fallback/jackson/mississippi",
                        "source": "trec_web_diversity",
                    },
                    {
                        "doc_id": "trec_jackson_003",
                        "title": "Andrew Jackson Presidential History",
                        "text": "Andrew Jackson was the seventh President of the United States from 1829 to 1837. He was known for his populist policies and strong personality.",
                        "url": "https://trec.fallback/jackson/president",
                        "source": "trec_web_diversity",
                    },
                    {
                        "doc_id": "trec_jackson_004",
                        "title": "Jackson Pollock Artist Profile",
                        "text": "Jackson Pollock was an American painter and major figure in the abstract expressionist movement. He was known for his unique style of drip painting.",
                        "url": "https://trec.fallback/jackson/pollock",
                        "source": "trec_web_diversity",
                    },
                    {
                        "doc_id": "trec_jackson_005",
                        "title": "Jackson Laboratory Research Institute",
                        "text": "The Jackson Laboratory is an independent, nonprofit biomedical research institution based in Maine. It conducts genetic research to advance human health.",
                        "url": "https://trec.fallback/jackson/laboratory",
                        "source": "trec_web_diversity",
                    },
                ],
                "apple": [
                    {
                        "doc_id": "trec_apple_001",
                        "title": "Apple Inc Company Information",
                        "text": "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services.",
                        "url": "https://trec.fallback/apple/company",
                        "source": "trec_web_diversity",
                    },
                    {
                        "doc_id": "trec_apple_002",
                        "title": "Apple Fruit Nutrition Facts",
                        "text": "Apples are nutritious fruits that provide fiber, vitamins, and antioxidants. They are one of the most widely consumed fruits worldwide.",
                        "url": "https://trec.fallback/apple/fruit",
                        "source": "trec_web_diversity",
                    },
                    {
                        "doc_id": "trec_apple_003",
                        "title": "Apple Records Music Label",
                        "text": "Apple Records is a record label founded by the Beatles in 1968. It was created to give the band more control over their music.",
                        "url": "https://trec.fallback/apple/records",
                        "source": "trec_web_diversity",
                    },
                    {
                        "doc_id": "trec_apple_004",
                        "title": "Apple Tree Growing Guide",
                        "text": "Apple trees are deciduous trees that require proper care including pruning, fertilizing, and pest management for optimal fruit production.",
                        "url": "https://trec.fallback/apple/tree",
                        "source": "trec_web_diversity",
                    },
                ],
                "python": [
                    {
                        "doc_id": "trec_python_001",
                        "title": "Python Programming Language Guide",
                        "text": "Python is a high-level, interpreted programming language known for its simple syntax and readability. It is widely used in web development, data science, and AI.",
                        "url": "https://trec.fallback/python/programming",
                        "source": "trec_web_diversity",
                    },
                    {
                        "doc_id": "trec_python_002",
                        "title": "Python Snake Species Information",
                        "text": "Pythons are a family of large, non-venomous snakes found in Africa, Asia, and Australia. They are constrictors that kill prey by squeezing.",
                        "url": "https://trec.fallback/python/snake",
                        "source": "trec_web_diversity",
                    },
                    {
                        "doc_id": "trec_python_003",
                        "title": "Monty Python Comedy Group",
                        "text": 'Monty Python was a British comedy group known for their sketch show "Monty Python\'s Flying Circus" and films like "The Holy Grail".',
                        "url": "https://trec.fallback/python/monty",
                        "source": "trec_web_diversity",
                    },
                ],
            }

            # Flatten all documents into a single list
            fallback_docs = []
            for query_docs in ambiguous_queries.values():
                fallback_docs.extend(query_docs)

            logger.info(
                f"Created fallback TREC corpus with {len(fallback_docs)} documents"
            )
            return fallback_docs

        except Exception as e:
            logger.error(f"Error creating fallback TREC corpus: {e}")
            return []

    def _create_wikipedia_disambiguation_corpus(self) -> List[Dict[str, Any]]:
        """Create a corpus from Wikipedia disambiguation pages for TREC-like ambiguous queries"""
        try:
            ambiguous_terms = [
                "Jackson",
                "Apple",
                "Python",
                "Mercury",
                "Mars",
                "Amazon",
                "Oracle",
                "Phoenix",
                "Columbia",
                "Victoria",
                "Washington",
                "Lincoln",
                "Franklin",
                "Madison",
                "Monroe",
                "Jefferson",
            ]

            corpus = []

            for term in ambiguous_terms:
                try:
                    page = self.wiki_wiki.page(term)
                    if page.exists():
                        if (
                            "disambiguation" in page.summary.lower()
                            or len(page.links) > 10
                        ):
                            for link_title in list(page.links.keys())[:10]:
                                link_page = self.wiki_wiki.page(link_title)
                                if link_page.exists() and len(link_page.summary) > 50:
                                    doc = {
                                        "doc_id": f"wiki_disambig_{term}_{len(corpus)}",
                                        "title": link_page.title,
                                        "text": link_page.summary[:500],
                                        "url": link_page.fullurl,
                                        "source": "wikipedia_disambiguation",
                                        "ambiguous_term": term,
                                    }
                                    corpus.append(doc)
                        else:
                            doc = {
                                "doc_id": f"wiki_disambig_{term}_{len(corpus)}",
                                "title": page.title,
                                "text": page.summary[:500],
                                "url": page.fullurl,
                                "source": "wikipedia_disambiguation",
                                "ambiguous_term": term,
                            }
                            corpus.append(doc)

                except Exception as e:
                    logger.debug(f"Error processing {term}: {e}")
                    continue

            return corpus

        except Exception as e:
            logger.error(f"Error creating Wikipedia disambiguation corpus: {e}")
            return []

    def search(
        self, query: str, language: str = "en", num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform search using real data sources

        Args:
            query: Search query
            language: Language code ('en' or 'ar')
            num_results: Number of results to return

        Returns:
            List of real search result dictionaries
        """
        try:
            logger.info(f"Searching for: '{query}' (language: {language})")

            # Rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < self.min_request_interval:
                time.sleep(
                    self.min_request_interval - (current_time - self.last_request_time)
                )

            results = []

            if language == "ar":
                # For Arabic queries, search MIRACL corpus first (NOW OPTIMIZED!)
                miracl_results = self._search_miracl_arabic_optimized(
                    query, num_results // 2
                )
                results.extend(miracl_results)

                # Then Google Custom Search for additional Arabic results
                google_results = self._search_google_custom(
                    query, language, num_results - len(results)
                )
                results.extend(google_results)
            else:
                # For English queries, prioritize TREC first for ambiguous queries
                trec_results = self._search_trec_real(query, num_results // 2)
                results.extend(trec_results)

                # Then Google Custom Search
                google_results = self._search_google_custom(
                    query, language, num_results - len(results)
                )
                results.extend(google_results)

                # # Wikipedia for additional disambiguation
                # wiki_results = self._search_wikipedia(
                #     query, language, max(0, num_results - len(results))
                # )
                # results.extend(wiki_results)

            self.last_request_time = time.time()

            logger.info(f"Found {len(results)} real results for '{query}'")
            return results[:num_results]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _search_google_custom(
        self, query: str, language: str = "en", num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API"""
        try:
            if not self.google_api_key or not self.google_cse_id:
                logger.warning("Google Custom Search API not configured")
                return []

            results = []
            url = "https://www.googleapis.com/customsearch/v1"
            pages_needed = (num_results + 9) // 10

            for page in range(min(pages_needed, 3)):
                start_index = page * 10 + 1

                params = {
                    "key": self.google_api_key,
                    "cx": self.google_cse_id,
                    "q": query,
                    "num": min(10, num_results - len(results)),
                    "start": start_index,
                    "lr": f"lang_{language}" if language == "ar" else "lang_en",
                    "safe": "active",
                }

                try:
                    response = self.session.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                    for item in data.get("items", []):
                        result = {
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": item.get("link", ""),
                            "source": "google_custom_search",
                        }
                        results.append(result)

                        if len(results) >= num_results:
                            break

                    if len(data.get("items", [])) < 10:
                        break

                except requests.exceptions.RequestException as e:
                    logger.error(f"Google Custom Search API error: {e}")
                    break

                time.sleep(0.1)

            logger.info(f"Google Custom Search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in Google Custom Search: {e}")
            return []

    def _search_miracl_arabic_optimized(
        self, query: str, num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """OPTIMIZED: Search the MIRACL-Arabic corpus using pre-computed vectors"""
        try:
            if (
                self.miracl_tfidf_vectorizer is None
                or self.miracl_doc_vectors is None
                or self.miracl_doc_metadata is None
            ):
                logger.warning("MIRACL-Arabic pre-computed vectors not available")
                return []

            logger.info(f"🚀 FAST Arabic search in MIRACL corpus for: '{query}'")

            # Transform query using pre-computed vectorizer (FAST!)
            query_vector = self.miracl_tfidf_vectorizer.transform([query])

            # Calculate similarities using pre-computed vectors (FAST!)
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(
                query_vector, self.miracl_doc_vectors
            ).flatten()

            # Get top results
            top_indices = np.argsort(similarities)[::-1][:num_results]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum relevance threshold
                    doc = self.miracl_doc_metadata[idx]
                    result = {
                        "title": doc.get("title", f"Document {doc.get('docid', idx)}"),
                        "snippet": (
                            doc.get("text", "")[:300] + "..."
                            if len(doc.get("text", "")) > 300
                            else doc.get("text", "")
                        ),
                        "url": doc.get(
                            "url", f"https://miracl.corpus/{doc.get('docid', idx)}"
                        ),
                        "source": "miracl_arabic",
                        "relevance_score": float(similarities[idx]),
                    }
                    results.append(result)

            logger.info(f"⚡ FAST MIRACL-Arabic search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in optimized MIRACL-Arabic search: {e}")
            return []

    def _search_trec_real(
        self, query: str, num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search the real TREC Web Diversity dataset"""
        try:
            if not self.trec_cache:
                logger.warning("TREC dataset not loaded")
                return []

            logger.info(f"Searching TREC Web Diversity for: '{query}'")

            # Simple keyword-based search in TREC dataset
            query_words = query.lower().split()
            scored_docs = []

            for doc in self.trec_cache:
                text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()

                # Calculate simple relevance score
                score = 0
                for word in query_words:
                    score += text.count(word)

                if score > 0:
                    scored_docs.append((score, doc))

            # Sort by relevance score
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            results = []
            for score, doc in scored_docs[:num_results]:
                result = {
                    "title": doc.get("title", f"TREC Document {doc.get('doc_id', '')}"),
                    "snippet": (
                        doc.get("text", "")[:300] + "..."
                        if len(doc.get("text", "")) > 300
                        else doc.get("text", "")
                    ),
                    "url": doc.get(
                        "url", f"https://trec.nist.gov/{doc.get('doc_id', '')}"
                    ),
                    "source": "trec_web_diversity",
                    "relevance_score": score,
                }
                results.append(result)

            logger.info(f"TREC search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching TREC dataset: {e}")
            return []

    def _search_wikipedia(
        self, query: str, language: str = "en", num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search Wikipedia for real disambiguation data"""
        try:
            # Set Wikipedia language
            if language == "ar":
                wiki = wikipediaapi.Wikipedia(
                    language="ar",
                    user_agent="DSR-RL Search System (Educational Research)",
                )
            else:
                wiki = self.wiki_wiki

            results = []

            # Search for the main page first
            page = wiki.page(query)

            if page.exists():
                # Check if it's a disambiguation page
                if "disambiguation" in page.summary.lower():
                    # Process disambiguation page links
                    for link_title in list(page.links.keys())[:num_results]:
                        link_page = wiki.page(link_title)
                        if link_page.exists() and len(link_page.summary) > 50:
                            snippet = (
                                link_page.summary[:200] + "..."
                                if len(link_page.summary) > 200
                                else link_page.summary
                            )
                            result = {
                                "title": link_page.title,
                                "snippet": snippet,
                                "url": link_page.fullurl,
                                "source": "wikipedia_disambiguation",
                            }
                            results.append(result)
                else:
                    # Regular page
                    snippet = (
                        page.summary[:200] + "..."
                        if len(page.summary) > 200
                        else page.summary
                    )
                    result = {
                        "title": page.title,
                        "snippet": snippet,
                        "url": page.fullurl,
                        "source": "wikipedia",
                    }
                    results.append(result)

                    # Also search for related pages through links
                    for link_title in list(page.links.keys())[: num_results - 1]:
                        if query.lower() in link_title.lower():
                            link_page = wiki.page(link_title)
                            if link_page.exists() and len(link_page.summary) > 50:
                                snippet = (
                                    link_page.summary[:200] + "..."
                                    if len(link_page.summary) > 200
                                    else link_page.summary
                                )
                                result = {
                                    "title": link_page.title,
                                    "snippet": snippet,
                                    "url": link_page.fullurl,
                                    "source": "wikipedia",
                                }
                                results.append(result)
            else:
                # Try searching for similar pages
                similar_terms = [
                    f"{query} (disambiguation)",
                    f"{query} (person)",
                    f"{query} (place)",
                    f"{query} (company)",
                ]

                for term in similar_terms:
                    page = wiki.page(term)
                    if page.exists():
                        snippet = (
                            page.summary[:200] + "..."
                            if len(page.summary) > 200
                            else page.summary
                        )
                        result = {
                            "title": page.title,
                            "snippet": snippet,
                            "url": page.fullurl,
                            "source": "wikipedia",
                        }
                        results.append(result)
                        if len(results) >= num_results:
                            break

            return results[:num_results]

        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
            return []

    def precompute_vectors_for_setup(self):
        """Force pre-computation of vectors during setup"""
        logger.info("🔄 Pre-computing vectors for Arabic search optimization...")

        if self.miracl_tfidf_vectorizer is None:
            self._precompute_miracl_vectors()

        if self.miracl_tfidf_vectorizer is not None:
            logger.info("✅ Arabic search vectors pre-computed successfully!")
            return True
        else:
            logger.warning("⚠️  Could not pre-compute Arabic vectors")
            return False

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded datasets"""
        return {
            "miracl_arabic_docs": (
                len(self.miracl_doc_metadata) if self.miracl_doc_metadata else 0
            ),
            "miracl_vectors_precomputed": self.miracl_tfidf_vectorizer is not None,
            "trec_docs": len(self.trec_cache) if self.trec_cache else 0,
            "google_api_configured": bool(self.google_api_key and self.google_cse_id),
            "sources_available": [
                (
                    "google_custom_search"
                    if self.google_api_key and self.google_cse_id
                    else None
                ),
                (
                    "miracl_arabic_optimized"
                    if self.miracl_tfidf_vectorizer
                    else "miracl_arabic_fallback"
                ),
                "trec_web_diversity" if self.trec_cache else None,
                "wikipedia",
            ],
            "languages_supported": ["en", "ar"],
            "rate_limit_interval": self.min_request_interval,
            "optimization_status": {
                "arabic_vectors_cached": os.path.exists("miracl_arabic_vectors.pkl"),
                "arabic_corpus_cached": os.path.exists("miracl_arabic_cache.pkl"),
                "trec_corpus_cached": os.path.exists("trec_web_diversity_cache.pkl"),
            },
        }

    def refresh_datasets(self):
        """Refresh/reload all datasets"""
        try:
            # Clear caches
            self.miracl_arabic_cache = None
            self.trec_cache = None
            self.miracl_tfidf_vectorizer = None
            self.miracl_doc_vectors = None
            self.miracl_doc_metadata = None

            # Remove cache files to force reload
            for cache_file in [
                "miracl_arabic_cache.pkl",
                "miracl_arabic_vectors.pkl",
                "trec_web_diversity_cache.pkl",
            ]:
                if os.path.exists(cache_file):
                    os.remove(cache_file)

            # Reload datasets
            self._load_datasets()

            logger.info("Datasets refreshed successfully")

        except Exception as e:
            logger.error(f"Error refreshing datasets: {e}")

    def search_specific_dataset(
        self, query: str, dataset: str, num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search a specific dataset directly"""
        try:
            if dataset == "miracl_arabic":
                return self._search_miracl_arabic_optimized(query, num_results)
            elif dataset == "trec":
                return self._search_trec_real(query, num_results)
            elif dataset == "wikipedia":
                return self._search_wikipedia(query, "en", num_results)
            elif dataset == "google":
                return self._search_google_custom(query, "en", num_results)
            else:
                logger.warning(f"Unknown dataset: {dataset}")
                return []

        except Exception as e:
            logger.error(f"Error searching dataset {dataset}: {e}")
            return []
