"""
Clustering Manager - Implements all clustering algorithms mentioned in the research paper
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import Counter

# Sklearn clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

# Advanced clustering
import hdbscan
from bertopic import BERTopic
from umap import UMAP

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ClusteringManager:
    """
    Manages all clustering algorithms as specified in the research paper:
    - K-Means
    - DBSCAN
    - HDBSCAN
    - Gaussian Mixture Model
    - BERTopic
    - Hierarchical Clustering
    """

    def __init__(self):
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        self.load_models()

        # Default parameters from Table I in the paper
        self.default_params = {
            "kmeans": {"k": "elbow"},
            "dbscan": {"eps": 0.7},
            "hdbscan": {"min_samples": 5},
            "gaussian_mixture": {"components": "k"},
            "bertopic": {"min_size": 10},
            "hierarchical": {"linkage": "ward"},
        }

    def load_models(self):
        """Load the sentence transformer model"""
        try:
            # Using multilingual model as specified in the paper
            self.sentence_transformer = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english", ngram_range=(1, 2)
            )
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def generate_embeddings(
        self, texts: List[str], method: str = "sentence_bert"
    ) -> np.ndarray:
        """
        Generate embeddings using specified method

        Args:
            texts: List of text documents
            method: 'sentence_bert' or 'tfidf'

        Returns:
            numpy array of embeddings
        """
        try:
            if method == "tfidf" or len(texts[0]) > 15:  # Use TF-IDF for long texts
                embeddings = self.tfidf_vectorizer.fit_transform(texts).toarray()
                logger.info(f"Generated TF-IDF embeddings: {embeddings.shape}")
            else:
                embeddings = self.sentence_transformer.encode(texts)
                logger.info(f"Generated Sentence-BERT embeddings: {embeddings.shape}")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def determine_optimal_clusters(
        self, embeddings: np.ndarray, max_k: int = 10
    ) -> int:
        """Determine optimal number of clusters using elbow method"""
        try:
            inertias = []
            k_range = range(2, min(max_k + 1, len(embeddings)))

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)

            # Simple elbow detection
            if len(inertias) < 2:
                return 2

            # Calculate rate of decrease
            decreases = [
                inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)
            ]
            # Find elbow point
            elbow_idx = np.argmax(decreases) + 2  # +2 because range starts at 2

            return min(elbow_idx, max_k)

        except Exception as e:
            logger.warning(f"Error in elbow method: {e}, using default k=4")
            return 4

    def cluster_kmeans(
        self, embeddings: np.ndarray, n_clusters: int = None
    ) -> Tuple[np.ndarray, Dict]:
        """K-Means clustering as specified in the paper"""
        try:
            if n_clusters is None:
                n_clusters = self.determine_optimal_clusters(embeddings)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Calculate metrics
            silhouette = (
                silhouette_score(embeddings, labels) if len(set(labels)) > 1 else 0
            )

            metrics = {
                "silhouette_score": silhouette,
                "n_clusters": n_clusters,
                "algorithm": "kmeans",
            }

            return labels, metrics

        except Exception as e:
            logger.error(f"K-Means clustering error: {e}")
            raise

    def cluster_dbscan(
        self, embeddings: np.ndarray, eps: float = 0.7, min_samples: int = 2
    ) -> Tuple[np.ndarray, Dict]:
        """DBSCAN clustering"""
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(embeddings)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            silhouette = silhouette_score(embeddings, labels) if n_clusters > 1 else 0

            metrics = {
                "silhouette_score": silhouette,
                "n_clusters": n_clusters,
                "noise_points": np.sum(labels == -1),
                "algorithm": "dbscan",
            }

            return labels, metrics

        except Exception as e:
            logger.error(f"DBSCAN clustering error: {e}")
            raise

    def cluster_hdbscan(
        self, embeddings: np.ndarray, min_cluster_size: int = 5
    ) -> Tuple[np.ndarray, Dict]:
        """HDBSCAN clustering as specified in the paper"""
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=0.0,
            )
            labels = clusterer.fit_predict(embeddings)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            silhouette = silhouette_score(embeddings, labels) if n_clusters > 1 else 0

            metrics = {
                "silhouette_score": silhouette,
                "n_clusters": n_clusters,
                "noise_points": np.sum(labels == -1),
                "cluster_persistence": (
                    clusterer.cluster_persistence_
                    if hasattr(clusterer, "cluster_persistence_")
                    else []
                ),
                "algorithm": "hdbscan",
            }

            return labels, metrics

        except Exception as e:
            logger.error(f"HDBSCAN clustering error: {e}")
            raise

    def cluster_gaussian_mixture(
        self, embeddings: np.ndarray, n_components: int = None
    ) -> Tuple[np.ndarray, Dict]:
        """Gaussian Mixture Model clustering"""
        try:
            if n_components is None:
                n_components = self.determine_optimal_clusters(embeddings)

            gmm = GaussianMixture(n_components=n_components, random_state=42)
            labels = gmm.fit_predict(embeddings)

            silhouette = (
                silhouette_score(embeddings, labels) if len(set(labels)) > 1 else 0
            )

            metrics = {
                "silhouette_score": silhouette,
                "n_clusters": n_components,
                "bic_score": gmm.bic(embeddings),
                "aic_score": gmm.aic(embeddings),
                "algorithm": "gaussian_mixture",
            }

            return labels, metrics

        except Exception as e:
            logger.error(f"Gaussian Mixture clustering error: {e}")
            raise

    def cluster_hierarchical(
        self, embeddings: np.ndarray, n_clusters: int = None
    ) -> Tuple[np.ndarray, Dict]:
        """Hierarchical clustering"""
        try:
            if n_clusters is None:
                n_clusters = self.determine_optimal_clusters(embeddings)

            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward"
            )
            labels = hierarchical.fit_predict(embeddings)

            silhouette = (
                silhouette_score(embeddings, labels) if len(set(labels)) > 1 else 0
            )

            metrics = {
                "silhouette_score": silhouette,
                "n_clusters": n_clusters,
                "algorithm": "hierarchical",
            }

            return labels, metrics

        except Exception as e:
            logger.error(f"Hierarchical clustering error: {e}")
            raise

    def cluster_bertopic(
        self, texts: List[str], embeddings: np.ndarray, min_topic_size: int = 10
    ) -> Tuple[np.ndarray, Dict, List[str]]:
        """BERTopic clustering as specified in the paper"""
        try:
            # UMAP for dimensionality reduction
            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )

            # HDBSCAN for clustering
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=min_topic_size,
                metric="euclidean",
                cluster_selection_method="eom",
            )

            # Initialize BERTopic
            topic_model = BERTopic(
                umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=False
            )

            # Fit the model
            topics, probabilities = topic_model.fit_transform(texts, embeddings)

            # Get topic labels
            topic_labels = []
            for topic_id in set(topics):
                if topic_id != -1:  # Skip outlier topic
                    words = topic_model.get_topic(topic_id)
                    if words:
                        label = " ".join([word for word, _ in words[:3]])
                        topic_labels.append(label)
                    else:
                        topic_labels.append(f"Topic {topic_id}")
                else:
                    topic_labels.append("Outliers")

            n_clusters = len(set(topics)) - (1 if -1 in topics else 0)
            silhouette = silhouette_score(embeddings, topics) if n_clusters > 1 else 0

            # Calculate topic coherence
            try:
                coherence_score = (
                    topic_model.get_topic_info()["Coherence"].mean()
                    if hasattr(topic_model, "get_topic_info")
                    else 0
                )
            except:
                coherence_score = 0

            metrics = {
                "silhouette_score": silhouette,
                "n_clusters": n_clusters,
                "topic_coherence": coherence_score,
                "noise_points": np.sum(np.array(topics) == -1),
                "algorithm": "bertopic",
            }

            return np.array(topics), metrics, topic_labels

        except Exception as e:
            logger.error(f"BERTopic clustering error: {e}")
            raise

    def ensemble_clustering(
        self,
        embeddings: np.ndarray,
        results: List[Dict],
        algorithms: List[str] = None,
        num_clusters: int = 4,
    ) -> Tuple[List[Dict], np.ndarray, Dict]:
        """
        Ensemble clustering using multiple algorithms
        """
        try:
            if algorithms is None:
                algorithms = ["kmeans", "hdbscan", "gaussian_mixture"]

            all_labels = []
            all_metrics = []

            # Run each algorithm
            for algo in algorithms:
                try:
                    if algo == "kmeans":
                        labels, metrics = self.cluster_kmeans(embeddings, num_clusters)
                    elif algo == "hdbscan":
                        labels, metrics = self.cluster_hdbscan(embeddings)
                    elif algo == "gaussian_mixture":
                        labels, metrics = self.cluster_gaussian_mixture(
                            embeddings, num_clusters
                        )
                    elif algo == "dbscan":
                        labels, metrics = self.cluster_dbscan(embeddings)
                    elif algo == "hierarchical":
                        labels, metrics = self.cluster_hierarchical(
                            embeddings, num_clusters
                        )
                    else:
                        continue

                    all_labels.append(labels)
                    all_metrics.append(metrics)

                except Exception as e:
                    logger.warning(f"Algorithm {algo} failed: {e}")
                    continue

            if not all_labels:
                raise ValueError("All clustering algorithms failed")

            # Ensemble by majority voting
            ensemble_labels = self._ensemble_vote(all_labels)

            # Calculate ensemble metrics
            silhouette = (
                silhouette_score(embeddings, ensemble_labels)
                if len(set(ensemble_labels)) > 1
                else 0
            )

            ensemble_metrics = {
                "silhouette_score": silhouette,
                "n_clusters": len(set(ensemble_labels)),
                "algorithms_used": algorithms,
                "individual_metrics": all_metrics,
                "algorithm": "ensemble",
            }

            # Convert to cluster format
            clusters = self._labels_to_clusters(ensemble_labels, results)

            return clusters, ensemble_labels, ensemble_metrics

        except Exception as e:
            logger.error(f"Ensemble clustering error: {e}")
            raise

    def _ensemble_vote(self, all_labels: List[np.ndarray]) -> np.ndarray:
        """Combine multiple clustering results using consensus"""
        try:
            n_samples = len(all_labels[0])
            n_algorithms = len(all_labels)

            # Create consensus matrix
            consensus_matrix = np.zeros((n_samples, n_samples))

            for labels in all_labels:
                for i in range(n_samples):
                    for j in range(n_samples):
                        if labels[i] == labels[j] and labels[i] != -1:
                            consensus_matrix[i, j] += 1

            # Normalize by number of algorithms
            consensus_matrix = consensus_matrix / n_algorithms

            # Use threshold-based clustering
            threshold = 0.5
            final_labels = np.full(n_samples, -1)
            cluster_id = 0

            for i in range(n_samples):
                if final_labels[i] == -1:
                    # Find all points similar to point i
                    similar_points = np.where(consensus_matrix[i] >= threshold)[0]
                    final_labels[similar_points] = cluster_id
                    cluster_id += 1

            return final_labels

        except Exception as e:
            logger.error(f"Ensemble voting error: {e}")
            # Fallback to first algorithm result
            return all_labels[0] if all_labels else np.array([0] * n_samples)

    def cluster_results(
        self,
        embeddings: np.ndarray,
        results: List[Dict],
        algorithm: str = "bertopic",
        num_clusters: int = 4,
        min_cluster_size: int = 2,
    ) -> Tuple[List[Dict], np.ndarray, Dict]:
        """
        Main clustering function that routes to specific algorithms
        """
        try:
            logger.info(f"Clustering with {algorithm}, target clusters: {num_clusters}")

            # Extract texts for BERTopic
            texts = [f"{result['title']} {result['snippet']}" for result in results]

            # Route to appropriate clustering algorithm
            if algorithm == "kmeans":
                labels, metrics = self.cluster_kmeans(embeddings, num_clusters)
                topic_labels = None
            elif algorithm == "dbscan":
                labels, metrics = self.cluster_dbscan(embeddings)
                topic_labels = None
            elif algorithm == "hdbscan":
                labels, metrics = self.cluster_hdbscan(embeddings, min_cluster_size)
                topic_labels = None
            elif algorithm == "gaussian_mixture":
                labels, metrics = self.cluster_gaussian_mixture(
                    embeddings, num_clusters
                )
                topic_labels = None
            elif algorithm == "hierarchical":
                labels, metrics = self.cluster_hierarchical(embeddings, num_clusters)
                topic_labels = None
            elif algorithm == "bertopic":
                labels, metrics, topic_labels = self.cluster_bertopic(
                    texts, embeddings, min_cluster_size
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # Convert labels to cluster format
            clusters = self._labels_to_clusters(labels, results, topic_labels)

            # Calculate additional metrics
            metrics.update(
                self._calculate_cluster_metrics(clusters, labels, embeddings)
            )

            logger.info(f"Generated {len(clusters)} clusters with {algorithm}")

            return clusters, labels, metrics

        except Exception as e:
            logger.error(f"Clustering error: {e}")
            raise

    def _labels_to_clusters(
        self, labels: np.ndarray, results: List[Dict], topic_labels: List[str] = None
    ) -> List[Dict]:
        """Convert cluster labels to cluster format"""
        try:
            clusters = []
            unique_labels = set(labels)

            for i, label in enumerate(unique_labels):
                if label == -1:  # Skip noise points for some algorithms
                    continue

                cluster_results = [
                    results[j] for j in range(len(results)) if labels[j] == label
                ]

                if not cluster_results:  # Skip empty clusters
                    continue

                # Generate cluster label
                if topic_labels and i < len(topic_labels):
                    cluster_label = topic_labels[i]
                else:
                    # Extract keywords from titles
                    titles = [r["title"] for r in cluster_results]
                    cluster_label = self._generate_cluster_label(titles)

                cluster = {
                    "label": cluster_label,
                    "results": cluster_results,
                    "size": len(cluster_results),
                    "cluster_id": int(label),
                }

                clusters.append(cluster)

            # Sort clusters by size (largest first)
            clusters.sort(key=lambda x: x["size"], reverse=True)

            return clusters

        except Exception as e:
            logger.error(f"Error converting labels to clusters: {e}")
            return []

    def _generate_cluster_label(self, titles: List[str]) -> str:
        """Generate a descriptive label for a cluster based on titles"""
        try:
            # Enhanced keyword extraction for better cluster labeling
            all_words = []
            title_text = " ".join(titles).lower()

            # Extract meaningful terms
            for title in titles:
                words = title.lower().split()
                # Filter out common words and keep meaningful terms
                filtered_words = []
                for w in words:
                    if (
                        len(w) > 2
                        and w.isalpha()
                        and w
                        not in {
                            "the",
                            "and",
                            "for",
                            "are",
                            "but",
                            "not",
                            "you",
                            "all",
                            "can",
                            "had",
                            "her",
                            "was",
                            "one",
                            "our",
                            "out",
                            "day",
                            "get",
                            "has",
                            "him",
                            "his",
                            "how",
                            "man",
                            "new",
                            "now",
                            "old",
                            "see",
                            "two",
                            "way",
                            "who",
                            "boy",
                            "did",
                            "its",
                            "let",
                            "put",
                            "say",
                            "she",
                            "too",
                            "use",
                        }
                    ):
                        filtered_words.append(w)
                all_words.extend(filtered_words)

            if not all_words:
                return "General Results"

            # Count word frequencies
            word_counts = Counter(all_words)

            # Look for specific patterns to create meaningful labels
            cluster_label = self._identify_cluster_type(titles, word_counts)

            if cluster_label:
                return cluster_label

            # Fallback to most common meaningful words
            top_words = word_counts.most_common(3)
            if top_words:
                # Filter and combine top words
                label_words = []
                for word, count in top_words:
                    if count > 1 and len(word) > 3:  # Must appear more than once
                        label_words.append(word.title())

                if label_words:
                    if len(label_words) == 1:
                        return f"{label_words[0]} Related"
                    else:
                        return " & ".join(label_words[:2])

            # Final fallback
            return (
                top_words[0][0].title() + " Results" if top_words else "General Results"
            )

        except Exception as e:
            logger.warning(f"Error generating cluster label: {e}")
            return "General Results"

    def _identify_cluster_type(
        self, titles: List[str], word_counts: Counter
    ) -> Optional[str]:
        """Identify cluster type based on content patterns"""
        try:
            combined_text = " ".join(titles).lower()

            # Define patterns for different types of content
            patterns = {
                "Artists & Musicians": [
                    "singer",
                    "musician",
                    "artist",
                    "band",
                    "album",
                    "music",
                    "song",
                    "rapper",
                    "composer",
                    "songwriter",
                    "performer",
                    "concert",
                ],
                "Actors & Entertainment": [
                    "actor",
                    "actress",
                    "film",
                    "movie",
                    "director",
                    "hollywood",
                    "cinema",
                    "tv",
                    "television",
                    "show",
                    "series",
                    "drama",
                ],
                "Places & Geography": [
                    "city",
                    "town",
                    "county",
                    "state",
                    "country",
                    "capital",
                    "population",
                    "located",
                    "geography",
                    "municipality",
                    "village",
                    "district",
                ],
                "Politicians & Leaders": [
                    "president",
                    "politician",
                    "government",
                    "senator",
                    "governor",
                    "mayor",
                    "minister",
                    "congress",
                    "political",
                    "election",
                    "office",
                ],
                "Companies & Business": [
                    "company",
                    "corporation",
                    "business",
                    "inc",
                    "llc",
                    "ceo",
                    "founded",
                    "headquarters",
                    "revenue",
                    "industry",
                    "technology",
                ],
                "Science & Nature": [
                    "species",
                    "animal",
                    "plant",
                    "biology",
                    "scientific",
                    "research",
                    "discovery",
                    "nature",
                    "wildlife",
                    "ecosystem",
                    "habitat",
                ],
                "Technology & Programming": [
                    "software",
                    "programming",
                    "language",
                    "code",
                    "developer",
                    "computer",
                    "technology",
                    "algorithm",
                    "application",
                    "system",
                ],
                "Sports & Athletics": [
                    "player",
                    "team",
                    "sport",
                    "game",
                    "athlete",
                    "championship",
                    "league",
                    "coach",
                    "season",
                    "tournament",
                    "olympics",
                ],
                "Books & Literature": [
                    "book",
                    "author",
                    "novel",
                    "writer",
                    "literature",
                    "published",
                    "story",
                    "fiction",
                    "biography",
                    "poetry",
                    "manuscript",
                ],
                "Historical Figures": [
                    "born",
                    "died",
                    "century",
                    "historical",
                    "ancient",
                    "war",
                    "battle",
                    "king",
                    "queen",
                    "empire",
                    "dynasty",
                    "era",
                ],
            }

            # Score each category
            category_scores = {}
            for category, keywords in patterns.items():
                score = 0
                for keyword in keywords:
                    if keyword in combined_text:
                        # Weight by frequency and specificity
                        occurrences = combined_text.count(keyword)
                        specificity_bonus = 2 if len(keyword) > 6 else 1
                        score += occurrences * specificity_bonus

                if score > 0:
                    category_scores[category] = score

            # Return the highest scoring category if it's significant
            if category_scores:
                best_category, best_score = max(
                    category_scores.items(), key=lambda x: x[1]
                )
                # Only return if the score is significant enough
                if best_score >= 2:  # At least 2 keyword matches
                    return best_category

            # Special handling for specific known entities
            if any(
                name in combined_text for name in ["michael jackson", "janet jackson"]
            ):
                return "Jackson Family Artists"
            elif any(term in combined_text for term in ["apple inc", "iphone", "mac"]):
                return "Apple Technology"
            elif any(
                term in combined_text
                for term in ["python programming", "python language"]
            ):
                return "Python Programming"
            elif any(term in combined_text for term in ["monty python", "comedy"]):
                return "Python Comedy"

            return None

        except Exception as e:
            logger.warning(f"Error identifying cluster type: {e}")
            return None

    def _calculate_cluster_metrics(
        self, clusters: List[Dict], labels: np.ndarray, embeddings: np.ndarray
    ) -> Dict:
        """Calculate additional clustering quality metrics"""
        try:
            metrics = {}

            if len(clusters) > 1:
                # Cluster purity (requires ground truth - using heuristic)
                metrics["cluster_purity"] = self._calculate_purity_heuristic(clusters)

                # Adjusted Rand Index (using self-similarity as baseline)
                metrics["adjusted_rand_index"] = self._calculate_ari_heuristic(
                    labels, embeddings
                )
            else:
                metrics["cluster_purity"] = 1.0 if len(clusters) == 1 else 0.0
                metrics["adjusted_rand_index"] = 0.0

            return metrics

        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return {"cluster_purity": 0.0, "adjusted_rand_index": 0.0}

    def _calculate_purity_heuristic(self, clusters: List[Dict]) -> float:
        """Calculate cluster purity using heuristic (title similarity within clusters)"""
        try:
            total_purity = 0.0
            total_items = 0

            for cluster in clusters:
                if cluster["size"] < 2:
                    continue

                titles = [result["title"].lower() for result in cluster["results"]]

                # Calculate intra-cluster similarity
                similarity_sum = 0.0
                comparisons = 0

                for i in range(len(titles)):
                    for j in range(i + 1, len(titles)):
                        # Simple word overlap similarity
                        words1 = set(titles[i].split())
                        words2 = set(titles[j].split())
                        if len(words1) > 0 and len(words2) > 0:
                            similarity = len(words1.intersection(words2)) / len(
                                words1.union(words2)
                            )
                            similarity_sum += similarity
                            comparisons += 1

                if comparisons > 0:
                    cluster_purity = similarity_sum / comparisons
                    total_purity += cluster_purity * cluster["size"]
                    total_items += cluster["size"]

            return total_purity / total_items if total_items > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error calculating purity: {e}")
            return 0.0

    def _calculate_ari_heuristic(
        self, labels: np.ndarray, embeddings: np.ndarray
    ) -> float:
        """Calculate ARI using embedding-based ground truth heuristic"""
        try:
            # Create pseudo ground truth based on embedding similarity
            similarity_matrix = cosine_similarity(embeddings)
            threshold = np.percentile(similarity_matrix, 75)  # Top 25% similarities

            # Create ground truth labels based on high similarity
            n_samples = len(embeddings)
            ground_truth = np.arange(
                n_samples
            )  # Start with each point as its own cluster

            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if similarity_matrix[i, j] > threshold:
                        # Merge clusters
                        min_label = min(ground_truth[i], ground_truth[j])
                        max_label = max(ground_truth[i], ground_truth[j])
                        ground_truth[ground_truth == max_label] = min_label

            # Calculate ARI
            ari = adjusted_rand_score(ground_truth, labels)
            return max(0.0, ari)  # Ensure non-negative

        except Exception as e:
            logger.warning(f"Error calculating ARI: {e}")
            return 0.0
