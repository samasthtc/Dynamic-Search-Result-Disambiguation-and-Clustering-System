"""
Clustering Engine
Handles clustering of real search results with multiple algorithms
"""

import logging
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
import hdbscan

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """
    Engine for clustering real search results using various algorithms
    """

    def __init__(self):
        self.algorithms = {
            "kmeans": self._kmeans_clustering,
            "hdbscan": self._hdbscan_clustering,
            "dbscan": self._dbscan_clustering,
            "adaptive": self._adaptive_clustering,
        }

        # Performance tracking
        self.algorithm_performance = defaultdict(list)

        logger.info("Clustering Engine initialized")

    def cluster(
        self,
        results: List[Dict],
        algorithm: str = "kmeans",
        num_clusters: int = 4,
        min_cluster_size: int = 2,
    ) -> List[Dict]:
        """
        Cluster search results using specified algorithm

        Args:
            results: List of search results with embeddings
            algorithm: Clustering algorithm to use
            num_clusters: Number of clusters (for applicable algorithms)
            min_cluster_size: Minimum cluster size

        Returns:
            List of clusters
        """
        if not results:
            return []

        logger.info(f"Clustering {len(results)} results with {algorithm}")

        try:
            # Extract embeddings
            embeddings = []
            valid_results = []

            for result in results:
                if result.get("embedding"):
                    embeddings.append(result["embedding"])
                    valid_results.append(result)

            if len(embeddings) < 2:
                logger.warning("Not enough valid embeddings for clustering")
                return self._create_single_cluster(results)

            embeddings = np.array(embeddings)

            # Apply clustering algorithm
            if algorithm in self.algorithms:
                cluster_func = self.algorithms[algorithm]
                labels = cluster_func(embeddings, num_clusters, min_cluster_size)
            else:
                logger.warning(f"Unknown algorithm {algorithm}, using kmeans")
                labels = self._kmeans_clustering(
                    embeddings, num_clusters, min_cluster_size
                )

            # Organize results into clusters
            clusters = self._organize_clusters(valid_results, labels)

            # Calculate quality score
            quality_score = self._calculate_clustering_quality(embeddings, labels)
            self.algorithm_performance[algorithm].append(quality_score)

            logger.info(
                f"Created {len(clusters)} clusters with quality score: {quality_score:.3f}"
            )

            return clusters

        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            return self._create_single_cluster(results)

    def _kmeans_clustering(
        self, embeddings: np.ndarray, num_clusters: int, min_cluster_size: int
    ) -> np.ndarray:
        """K-means clustering"""
        try:
            # Adjust number of clusters if needed
            n_samples = len(embeddings)
            actual_clusters = min(num_clusters, n_samples)

            if actual_clusters < 2:
                return np.zeros(n_samples)

            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)

            labels = kmeans.fit_predict(embeddings)

            # Merge small clusters
            return self._merge_small_clusters(labels, min_cluster_size)

        except Exception as e:
            logger.error(f"K-means error: {str(e)}")
            return np.zeros(len(embeddings))

    def _hdbscan_clustering(
        self, embeddings: np.ndarray, num_clusters: int, min_cluster_size: int
    ) -> np.ndarray:
        """HDBSCAN clustering"""
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(min_cluster_size, 2),
                min_samples=1,
                metric="euclidean",
            )

            labels = clusterer.fit_predict(embeddings)

            # Handle noise points (-1 labels)
            return self._handle_noise_points(embeddings, labels)

        except Exception as e:
            logger.error(f"HDBSCAN error: {str(e)}")
            return np.zeros(len(embeddings))

    def _dbscan_clustering(
        self, embeddings: np.ndarray, num_clusters: int, min_cluster_size: int
    ) -> np.ndarray:
        """DBSCAN clustering"""
        try:
            # Estimate epsilon parameter
            eps = self._estimate_eps(embeddings)

            dbscan = DBSCAN(eps=eps, min_samples=min_cluster_size)

            labels = dbscan.fit_predict(embeddings)

            # Handle noise points
            return self._handle_noise_points(embeddings, labels)

        except Exception as e:
            logger.error(f"DBSCAN error: {str(e)}")
            return np.zeros(len(embeddings))

    def _adaptive_clustering(
        self, embeddings: np.ndarray, num_clusters: int, min_cluster_size: int
    ) -> np.ndarray:
        """Adaptive clustering that chooses best algorithm"""
        try:
            n_samples = len(embeddings)

            # Choose algorithm based on data characteristics
            if n_samples < 10:
                # Small dataset: use K-means
                return self._kmeans_clustering(
                    embeddings, num_clusters, min_cluster_size
                )
            elif n_samples > 100:
                # Large dataset: use HDBSCAN
                return self._hdbscan_clustering(
                    embeddings, num_clusters, min_cluster_size
                )
            else:
                # Medium dataset: choose based on performance history
                best_algorithm = self._get_best_algorithm()

                if best_algorithm == "hdbscan":
                    return self._hdbscan_clustering(
                        embeddings, num_clusters, min_cluster_size
                    )
                else:
                    return self._kmeans_clustering(
                        embeddings, num_clusters, min_cluster_size
                    )

        except Exception as e:
            logger.error(f"Adaptive clustering error: {str(e)}")
            return self._kmeans_clustering(embeddings, num_clusters, min_cluster_size)

    def _get_best_algorithm(self) -> str:
        """Get best performing algorithm based on history"""
        avg_scores = {}

        for alg, scores in self.algorithm_performance.items():
            if scores:
                avg_scores[alg] = np.mean(scores[-5:])  # Last 5 scores

        if not avg_scores:
            return "kmeans"

        return max(avg_scores.items(), key=lambda x: x[1])[0]

    def _estimate_eps(self, embeddings: np.ndarray) -> float:
        """Estimate epsilon parameter for DBSCAN"""
        try:
            from sklearn.neighbors import NearestNeighbors

            k = min(4, len(embeddings) - 1)
            if k < 1:
                return 0.5

            nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
            distances, _ = nbrs.kneighbors(embeddings)

            # Use 90th percentile of k-distances
            k_distances = np.sort(distances[:, -1])
            eps = k_distances[int(0.9 * len(k_distances))]

            return max(eps, 0.1)  # Minimum epsilon

        except Exception as e:
            logger.error(f"Epsilon estimation error: {str(e)}")
            return 0.5

    def _merge_small_clusters(self, labels: np.ndarray, min_size: int) -> np.ndarray:
        """Merge clusters smaller than minimum size"""
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Find clusters to merge
        small_clusters = unique_labels[counts < min_size]

        if len(small_clusters) == 0:
            return labels

        # Find largest cluster to merge into
        large_clusters = unique_labels[counts >= min_size]

        if len(large_clusters) == 0:
            # All clusters are small, keep as is
            return labels

        target_cluster = large_clusters[
            np.argmax(counts[np.isin(unique_labels, large_clusters)])
        ]

        # Merge small clusters
        merged_labels = labels.copy()
        for small_cluster in small_clusters:
            merged_labels[labels == small_cluster] = target_cluster

        return merged_labels

    def _handle_noise_points(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Assign noise points to nearest clusters"""
        noise_mask = labels == -1

        if not np.any(noise_mask):
            return labels

        # Get non-noise points
        non_noise_mask = ~noise_mask
        non_noise_embeddings = embeddings[non_noise_mask]
        non_noise_labels = labels[non_noise_mask]

        if len(non_noise_embeddings) == 0:
            # All points are noise, assign to single cluster
            return np.zeros_like(labels)

        # Assign noise points to nearest non-noise cluster
        try:
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=1).fit(non_noise_embeddings)
            noise_embeddings = embeddings[noise_mask]
            _, indices = nbrs.kneighbors(noise_embeddings)

            new_labels = labels.copy()
            new_labels[noise_mask] = non_noise_labels[indices.flatten()]

            return new_labels

        except Exception as e:
            logger.error(f"Noise handling error: {str(e)}")
            # Fallback: assign all noise to cluster 0
            new_labels = labels.copy()
            new_labels[noise_mask] = 0
            return new_labels

    def _organize_clusters(self, results: List[Dict], labels: np.ndarray) -> List[Dict]:
        """Organize results into cluster structures"""
        cluster_dict = defaultdict(list)

        for i, label in enumerate(labels):
            cluster_dict[int(label)].append(results[i])

        clusters = []
        for cluster_id, cluster_results in cluster_dict.items():
            if len(cluster_results) > 0:
                cluster = {
                    "id": cluster_id,
                    "label": self._generate_cluster_label(cluster_results),
                    "results": cluster_results,
                    "size": len(cluster_results),
                    "coherence_score": self._calculate_cluster_coherence(
                        cluster_results
                    ),
                    "diversity_score": self._calculate_cluster_diversity(
                        cluster_results
                    ),
                    "sources": list(
                        set(r.get("dataset_source", "unknown") for r in cluster_results)
                    ),
                    "categories": list(
                        set(r.get("category", "general") for r in cluster_results)
                    ),
                }
                clusters.append(cluster)

        return sorted(clusters, key=lambda x: x["size"], reverse=True)

    def _generate_cluster_label(self, results: List[Dict]) -> str:
        """Generate descriptive label for cluster"""
        # Analyze sources
        sources = [r.get("dataset_source", "") for r in results]
        source_counts = defaultdict(int)
        for source in sources:
            source_counts[source] += 1

        # Analyze categories
        categories = [r.get("category", "") for r in results]
        category_counts = defaultdict(int)
        for category in categories:
            category_counts[category] += 1

        # Get most common source and category
        most_common_source = (
            max(source_counts.items(), key=lambda x: x[1])[0] if source_counts else ""
        )
        most_common_category = (
            max(category_counts.items(), key=lambda x: x[1])[0]
            if category_counts
            else ""
        )

        # Generate label
        if "wikipedia" in most_common_source:
            return f"Wikipedia: {most_common_category.replace('_', ' ').title()}"
        elif "arxiv" in most_common_source:
            return f"Academic Papers: {most_common_category.replace('_', ' ').title()}"
        elif most_common_category:
            return most_common_category.replace("_", " ").title()
        else:
            return "Mixed Content"

    def _calculate_cluster_coherence(self, results: List[Dict]) -> float:
        """Calculate coherence score for cluster"""
        try:
            if len(results) < 2:
                return 1.0

            embeddings = [r.get("embedding", []) for r in results if r.get("embedding")]

            if len(embeddings) < 2:
                return 0.5

            embeddings = np.array(embeddings)
            centroid = np.mean(embeddings, axis=0)

            # Calculate average cosine similarity to centroid
            similarities = []
            for embedding in embeddings:
                sim = np.dot(embedding, centroid) / (
                    np.linalg.norm(embedding) * np.linalg.norm(centroid)
                )
                similarities.append(max(0, sim))

            return np.mean(similarities)

        except Exception as e:
            logger.error(f"Coherence calculation error: {str(e)}")
            return 0.5

    def _calculate_cluster_diversity(self, results: List[Dict]) -> float:
        """Calculate diversity score for cluster"""
        try:
            if len(results) <= 1:
                return 0.0

            # Source diversity
            sources = set(r.get("dataset_source", "") for r in results)
            source_diversity = len(sources) / len(results)

            # Category diversity
            categories = set(r.get("category", "") for r in results)
            category_diversity = len(categories) / len(results)

            return (source_diversity + category_diversity) / 2

        except Exception as e:
            logger.error(f"Diversity calculation error: {str(e)}")
            return 0.5

    def _create_single_cluster(self, results: List[Dict]) -> List[Dict]:
        """Create single cluster containing all results"""
        if not results:
            return []

        return [
            {
                "id": 0,
                "label": "All Results",
                "results": results,
                "size": len(results),
                "coherence_score": 0.5,
                "diversity_score": 1.0,
                "sources": list(
                    set(r.get("dataset_source", "unknown") for r in results)
                ),
                "categories": list(set(r.get("category", "general") for r in results)),
            }
        ]

    def _calculate_clustering_quality(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> float:
        """Calculate overall clustering quality"""
        try:
            unique_labels = np.unique(labels)

            if len(unique_labels) < 2:
                return 0.0

            # Silhouette score
            silhouette = silhouette_score(embeddings, labels)

            # Cluster balance (prefer balanced clusters)
            _, counts = np.unique(labels, return_counts=True)
            balance = (
                1.0 - (np.std(counts) / np.mean(counts)) if len(counts) > 1 else 1.0
            )
            balance = max(0.0, balance)

            # Combined score
            quality = silhouette * 0.7 + balance * 0.3
            return max(0.0, quality)

        except Exception as e:
            logger.error(f"Quality calculation error: {str(e)}")
            return 0.0

    def calculate_metrics(self, results: List[Dict], clusters: List[Dict]) -> Dict:
        """Calculate comprehensive clustering metrics"""
        try:
            if not results or not clusters:
                return {
                    "silhouette_score": 0.0,
                    "cluster_purity": 0.0,
                    "num_clusters": 0,
                    "avg_cluster_size": 0.0,
                }

            # Extract data for metrics
            cluster_labels = []
            true_categories = []

            for cluster in clusters:
                for result in cluster["results"]:
                    cluster_labels.append(cluster["id"])
                    true_categories.append(result.get("category", "general"))

            # Calculate metrics
            metrics = {
                "num_clusters": len(clusters),
                "avg_cluster_size": np.mean([c["size"] for c in clusters]),
                "cluster_purity": self._calculate_purity(
                    cluster_labels, true_categories
                ),
                "silhouette_score": 0.0,  # Will be calculated if embeddings available
                "coherence_scores": [c["coherence_score"] for c in clusters],
                "diversity_scores": [c["diversity_score"] for c in clusters],
            }

            # Calculate silhouette score if embeddings available
            embeddings = [r.get("embedding", []) for r in results if r.get("embedding")]
            if len(embeddings) > 1 and len(set(cluster_labels)) > 1:
                try:
                    embeddings = np.array(embeddings)
                    metrics["silhouette_score"] = silhouette_score(
                        embeddings, cluster_labels[: len(embeddings)]
                    )
                except:
                    pass

            return metrics

        except Exception as e:
            logger.error(f"Metrics calculation error: {str(e)}")
            return {"error": str(e)}

    def _calculate_purity(
        self, cluster_labels: List[int], true_labels: List[str]
    ) -> float:
        """Calculate cluster purity"""
        try:
            if len(cluster_labels) == 0:
                return 0.0

            clusters = defaultdict(list)
            for i, cluster_id in enumerate(cluster_labels):
                clusters[cluster_id].append(true_labels[i])

            total_purity = 0
            for cluster_items in clusters.values():
                # Most common true label in this cluster
                label_counts = defaultdict(int)
                for label in cluster_items:
                    label_counts[label] += 1

                max_count = max(label_counts.values()) if label_counts else 0
                total_purity += max_count

            return total_purity / len(cluster_labels)

        except Exception as e:
            logger.error(f"Purity calculation error: {str(e)}")
            return 0.0
