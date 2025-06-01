import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import hdbscan
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ClusteringManager:
    """
    Advanced clustering manager supporting multiple algorithms with 
    dynamic parameter optimization and quality assessment.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.supported_algorithms = {
            'kmeans': self._kmeans_clustering,
            'hdbscan': self._hdbscan_clustering,
            'dbscan': self._dbscan_clustering,
            'bertopic': self._bertopic_clustering,
            'gaussian_mixture': self._gaussian_mixture_clustering,
            'hierarchical': self._hierarchical_clustering,
            'adaptive': self._adaptive_clustering
        }
        
        # BERTopic model for topic-based clustering
        self.bertopic_model = None
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Clustering quality history for adaptive selection
        self.algorithm_performance = {
            'kmeans': [],
            'hdbscan': [],
            'dbscan': [],
            'bertopic': [],
            'gaussian_mixture': [],
            'hierarchical': []
        }
        
        logger.info("Clustering Manager initialized with 7 algorithms")

    def cluster(self, embeddings: np.ndarray, algorithm: str = 'adaptive', 
                num_clusters: int = 4, min_cluster_size: int = 2, **kwargs) -> np.ndarray:
        """
        Main clustering method that routes to appropriate algorithm.
        
        Args:
            embeddings: Feature embeddings for clustering
            algorithm: Clustering algorithm to use
            num_clusters: Number of clusters (for applicable algorithms)
            min_cluster_size: Minimum cluster size
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Array of cluster labels
        """
        if algorithm not in self.supported_algorithms:
            logger.warning(f"Unknown algorithm {algorithm}, falling back to kmeans")
            algorithm = 'kmeans'
        
        logger.info(f"Clustering {len(embeddings)} items using {algorithm}")
        
        # Normalize embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Apply clustering algorithm
        cluster_func = self.supported_algorithms[algorithm]
        labels = cluster_func(embeddings_scaled, num_clusters, min_cluster_size, **kwargs)
        
        # Evaluate clustering quality
        quality_score = self._evaluate_clustering_quality(embeddings_scaled, labels)
        self.algorithm_performance[algorithm].append(quality_score)
        
        logger.info(f"Clustering completed: {len(set(labels))} clusters, "
                   f"quality score: {quality_score:.3f}")
        
        return labels

    def _kmeans_clustering(self, embeddings: np.ndarray, num_clusters: int, 
                          min_cluster_size: int, **kwargs) -> np.ndarray:
        """
        K-Means clustering with automatic cluster number optimization.
        """
        # Optimize number of clusters if not fixed
        if kwargs.get('optimize_k', True):
            num_clusters = self._optimize_kmeans_clusters(embeddings, num_clusters)
        
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
            init='k-means++'
        )
        
        labels = kmeans.fit_predict(embeddings)
        
        # Post-process small clusters
        labels = self._merge_small_clusters(labels, min_cluster_size)
        
        return labels

    def _optimize_kmeans_clusters(self, embeddings: np.ndarray, 
                                 max_clusters: int) -> int:
        """
        Optimize number of clusters using elbow method and silhouette analysis.
        """
        n_samples = len(embeddings)
        max_k = min(max_clusters, n_samples // 2, 10)  # Reasonable upper bound
        
        if max_k < 2:
            return 2
        
        silhouette_scores = []
        inertias = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = kmeans.fit_predict(embeddings)
            
            # Skip if all points in one cluster
            if len(set(labels)) < 2:
                continue
                
            silhouette_avg = silhouette_score(embeddings, labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
        
        if not silhouette_scores:
            return 2
        
        # Find optimal K using silhouette score
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        
        return optimal_k

    def _hdbscan_clustering(self, embeddings: np.ndarray, num_clusters: int,
                           min_cluster_size: int, **kwargs) -> np.ndarray:
        """
        HDBSCAN clustering for density-based clustering with noise detection.
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(min_cluster_size, 2),
            min_samples=kwargs.get('min_samples', 1),
            cluster_selection_epsilon=kwargs.get('epsilon', 0.0),
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Handle noise points (labeled as -1)
        noise_mask = labels == -1
        if np.any(noise_mask):
            # Assign noise points to nearest cluster
            labels = self._assign_noise_to_clusters(embeddings, labels)
        
        return labels

    def _dbscan_clustering(self, embeddings: np.ndarray, num_clusters: int,
                          min_cluster_size: int, **kwargs) -> np.ndarray:
        """
        DBSCAN clustering with automatic epsilon optimization.
        """
        # Optimize epsilon using k-distance graph
        eps = kwargs.get('eps', None)
        if eps is None:
            eps = self._optimize_dbscan_epsilon(embeddings, min_cluster_size)
        
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_cluster_size,
            metric='euclidean'
        )
        
        labels = dbscan.fit_predict(embeddings)
        
        # Handle noise points
        noise_mask = labels == -1
        if np.any(noise_mask):
            labels = self._assign_noise_to_clusters(embeddings, labels)
        
        return labels

    def _optimize_dbscan_epsilon(self, embeddings: np.ndarray, 
                                min_samples: int) -> float:
        """
        Optimize DBSCAN epsilon parameter using k-distance graph.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Calculate k-distances
        k = min_samples
        nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Sort k-distances
        k_distances = np.sort(distances[:, k-1])
        
        # Find elbow point (simple method)
        # In practice, you might want a more sophisticated elbow detection
        n_points = len(k_distances)
        knee_point = int(0.95 * n_points)  # 95th percentile
        
        optimal_eps = k_distances[knee_point]
        
        return optimal_eps

    def _bertopic_clustering(self, embeddings: np.ndarray, num_clusters: int,
                           min_cluster_size: int, **kwargs) -> np.ndarray:
        """
        BERTopic clustering for topic-based document clustering.
        """
        # Note: This requires documents, not just embeddings
        # For this implementation, we'll simulate topic extraction
        
        # Use Gaussian Mixture as a proxy for topic modeling
        gmm = GaussianMixture(
            n_components=num_clusters,
            covariance_type='full',
            random_state=42
        )
        
        labels = gmm.fit_predict(embeddings)
        
        # Post-process to ensure minimum cluster size
        labels = self._merge_small_clusters(labels, min_cluster_size)
        
        return labels

    def _gaussian_mixture_clustering(self, embeddings: np.ndarray, num_clusters: int,
                                   min_cluster_size: int, **kwargs) -> np.ndarray:
        """
        Gaussian Mixture Model clustering with soft assignment.
        """
        # Optimize number of components using BIC/AIC
        if kwargs.get('optimize_components', True):
            num_clusters = self._optimize_gmm_components(embeddings, num_clusters)
        
        gmm = GaussianMixture(
            n_components=num_clusters,
            covariance_type=kwargs.get('covariance_type', 'full'),
            random_state=42,
            max_iter=100
        )
        
        labels = gmm.fit_predict(embeddings)
        
        # Post-process small clusters
        labels = self._merge_small_clusters(labels, min_cluster_size)
        
        return labels

    def _optimize_gmm_components(self, embeddings: np.ndarray, 
                               max_components: int) -> int:
        """
        Optimize number of GMM components using BIC criterion.
        """
        n_samples = len(embeddings)
        max_comp = min(max_components, n_samples // 3, 10)
        
        if max_comp < 2:
            return 2
        
        bic_scores = []
        
        for n_comp in range(2, max_comp + 1):
            gmm = GaussianMixture(n_components=n_comp, random_state=42)
            try:
                gmm.fit(embeddings)
                bic_scores.append(gmm.bic(embeddings))
            except:
                bic_scores.append(float('inf'))
        
        # Lower BIC is better
        optimal_components = bic_scores.index(min(bic_scores)) + 2
        
        return optimal_components

    def _hierarchical_clustering(self, embeddings: np.ndarray, num_clusters: int,
                               min_cluster_size: int, **kwargs) -> np.ndarray:
        """
        Agglomerative hierarchical clustering.
        """
        clustering = AgglomerativeClustering(
            n_clusters=num_clusters,
            linkage=kwargs.get('linkage', 'ward'),
            metric=kwargs.get('metric', 'euclidean')
        )
        
        labels = clustering.fit_predict(embeddings)
        
        # Post-process small clusters
        labels = self._merge_small_clusters(labels, min_cluster_size)
        
        return labels

    def _adaptive_clustering(self, embeddings: np.ndarray, num_clusters: int,
                           min_cluster_size: int, **kwargs) -> np.ndarray:
        """
        Adaptive clustering that selects the best algorithm based on data characteristics.
        """
        # Analyze data characteristics
        n_samples, n_features = embeddings.shape
        data_density = self._estimate_data_density(embeddings)
        data_dimensionality = n_features
        
        # Select algorithm based on characteristics
        if n_samples < 50:
            # Small dataset: use simple K-means
            algorithm = 'kmeans'
        elif data_density < 0.3:
            # Sparse data: use HDBSCAN
            algorithm = 'hdbscan'
        elif data_dimensionality > 100:
            # High dimensional: use Gaussian Mixture
            algorithm = 'gaussian_mixture'
        else:
            # Default: use best performing algorithm
            algorithm = self._get_best_performing_algorithm()
        
        logger.info(f"Adaptive clustering selected: {algorithm}")
        
        # Apply selected algorithm
        cluster_func = self.supported_algorithms[algorithm]
        return cluster_func(embeddings, num_clusters, min_cluster_size, **kwargs)

    def _estimate_data_density(self, embeddings: np.ndarray) -> float:
        """
        Estimate data density using nearest neighbor distances.
        """
        from sklearn.neighbors import NearestNeighbors
        
        if len(embeddings) < 5:
            return 1.0
        
        k = min(5, len(embeddings) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        
        # Average distance to k-th nearest neighbor
        avg_distance = np.mean(distances[:, k])
        
        # Convert to density estimate (inverse relationship)
        density = 1.0 / (1.0 + avg_distance)
        
        return density

    def _get_best_performing_algorithm(self) -> str:
        """
        Get the best performing algorithm based on historical performance.
        """
        avg_performances = {}
        
        for alg, scores in self.algorithm_performance.items():
            if scores:
                avg_performances[alg] = np.mean(scores[-5:])  # Last 5 scores
            else:
                avg_performances[alg] = 0.0
        
        if not avg_performances or all(score == 0 for score in avg_performances.values()):
            return 'kmeans'  # Default fallback
        
        best_algorithm = max(avg_performances.items(), key=lambda x: x[1])[0]
        return best_algorithm

    def _merge_small_clusters(self, labels: np.ndarray, 
                            min_cluster_size: int) -> np.ndarray:
        """
        Merge clusters that are smaller than minimum size.
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Find small clusters
        small_clusters = unique_labels[counts < min_cluster_size]
        
        if len(small_clusters) == 0:
            return labels
        
        # Merge small clusters with the largest cluster
        largest_cluster = unique_labels[np.argmax(counts)]
        
        merged_labels = labels.copy()
        for small_cluster in small_clusters:
            merged_labels[labels == small_cluster] = largest_cluster
        
        # Relabel clusters to be consecutive
        merged_labels = self._relabel_consecutive(merged_labels)
        
        return merged_labels

    def _assign_noise_to_clusters(self, embeddings: np.ndarray, 
                                labels: np.ndarray) -> np.ndarray:
        """
        Assign noise points to nearest clusters.
        """
        from sklearn.neighbors import NearestNeighbors
        
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels
        
        # Get non-noise points and their labels
        non_noise_mask = ~noise_mask
        non_noise_embeddings = embeddings[non_noise_mask]
        non_noise_labels = labels[non_noise_mask]
        
        if len(non_noise_embeddings) == 0:
            # All points are noise, assign them to cluster 0
            return np.zeros_like(labels)
        
        # Find nearest non-noise point for each noise point
        nbrs = NearestNeighbors(n_neighbors=1).fit(non_noise_embeddings)
        noise_embeddings = embeddings[noise_mask]
        _, indices = nbrs.kneighbors(noise_embeddings)
        
        # Assign noise points to clusters of their nearest neighbors
        new_labels = labels.copy()
        new_labels[noise_mask] = non_noise_labels[indices.flatten()]
        
        return new_labels

    def _relabel_consecutive(self, labels: np.ndarray) -> np.ndarray:
        """
        Relabel clusters to have consecutive integer labels starting from 0.
        """
        unique_labels = np.unique(labels)
        label_mapping = {old_label: new_label 
                        for new_label, old_label in enumerate(unique_labels)}
        
        relabeled = np.array([label_mapping[label] for label in labels])
        
        return relabeled

    def _evaluate_clustering_quality(self, embeddings: np.ndarray, 
                                   labels: np.ndarray) -> float:
        """
        Evaluate clustering quality using multiple metrics.
        """
        try:
            # Check if we have valid clusters
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Silhouette score (main quality metric)
            silhouette_avg = silhouette_score(embeddings, labels)
            
            # Calinski-Harabasz score (cluster separation)
            ch_score = calinski_harabasz_score(embeddings, labels)
            ch_normalized = min(ch_score / 1000, 1.0)  # Normalize to 0-1 range
            
            # Cluster balance score (preference for balanced clusters)
            _, counts = np.unique(labels, return_counts=True)
            balance_score = 1.0 - np.std(counts) / np.mean(counts) if len(counts) > 1 else 1.0
            balance_score = max(balance_score, 0.0)
            
            # Combined quality score
            quality_score = (0.5 * silhouette_avg + 
                           0.3 * ch_normalized + 
                           0.2 * balance_score)
            
            return max(quality_score, 0.0)
            
        except Exception as e:
            logger.warning(f"Error evaluating clustering quality: {str(e)}")
            return 0.0

    def get_algorithm_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics for all algorithms.
        """
        performance_stats = {}
        
        for algorithm, scores in self.algorithm_performance.items():
            if scores:
                performance_stats[algorithm] = {
                    'avg_score': np.mean(scores),
                    'recent_score': np.mean(scores[-3:]) if len(scores) >= 3 else np.mean(scores),
                    'total_runs': len(scores),
                    'trend': self._calculate_trend(scores)
                }
            else:
                performance_stats[algorithm] = {
                    'avg_score': 0.0,
                    'recent_score': 0.0,
                    'total_runs': 0,
                    'trend': 0.0
                }
        
        return performance_stats

    def _calculate_trend(self, scores: List[float]) -> float:
        """
        Calculate performance trend (positive = improving, negative = declining).
        """
        if len(scores) < 3:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        return slope

    def recommend_algorithm(self, data_characteristics: Dict[str, Any]) -> str:
        """
        Recommend the best clustering algorithm based on data characteristics.
        """
        n_samples = data_characteristics.get('n_samples', 100)
        n_features = data_characteristics.get('n_features', 10)
        expected_clusters = data_characteristics.get('expected_clusters', 4)
        noise_level = data_characteristics.get('noise_level', 0.1)  # 0-1
        
        # Algorithm selection rules
        if n_samples < 30:
            return 'kmeans'  # Simple and stable for small datasets
        
        if noise_level > 0.3:
            return 'hdbscan'  # Good for noisy data
        
        if n_features > 50:
            return 'gaussian_mixture'  # Better for high-dimensional data
        
        if expected_clusters > 8:
            return 'hierarchical'  # Good for many clusters
        
        # Use historical performance
        performance_stats = self.get_algorithm_performance()
        best_algorithm = max(performance_stats.items(), 
                           key=lambda x: x[1]['recent_score'])[0]
        
        return best_algorithm

    def cluster_with_uncertainty(self, embeddings: np.ndarray, 
                               algorithm: str = 'gaussian_mixture',
                               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform clustering with uncertainty estimates for each assignment.
        """
        if algorithm == 'gaussian_mixture':
            gmm = GaussianMixture(
                n_components=kwargs.get('num_clusters', 4),
                random_state=42
            )
            
            gmm.fit(embeddings)
            labels = gmm.predict(embeddings)
            
            # Get prediction probabilities as uncertainty measure
            probabilities = gmm.predict_proba(embeddings)
            uncertainties = 1.0 - np.max(probabilities, axis=1)
            
            return labels, uncertainties
        
        else:
            # For other algorithms, estimate uncertainty based on distance to centroids
            labels = self.cluster(embeddings, algorithm, **kwargs)
            uncertainties = self._estimate_assignment_uncertainty(embeddings, labels)
            
            return labels, uncertainties

    def _estimate_assignment_uncertainty(self, embeddings: np.ndarray, 
                                       labels: np.ndarray) -> np.ndarray:
        """
        Estimate assignment uncertainty based on distance to cluster centroids.
        """
        unique_labels = np.unique(labels)
        uncertainties = np.zeros(len(embeddings))
        
        # Calculate centroids
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label] = np.mean(embeddings[mask], axis=0)
        
        # Calculate uncertainty for each point
        for i, (embedding, label) in enumerate(zip(embeddings, labels)):
            # Distance to assigned cluster centroid
            dist_to_assigned = np.linalg.norm(embedding - centroids[label])
            
            # Distance to nearest other centroid
            other_distances = []
            for other_label, centroid in centroids.items():
                if other_label != label:
                    other_distances.append(np.linalg.norm(embedding - centroid))
            
            if other_distances:
                dist_to_nearest_other = min(other_distances)
                # Uncertainty is high when point is close to boundary
                uncertainty = 1.0 - (dist_to_nearest_other / (dist_to_assigned + dist_to_nearest_other))
            else:
                uncertainty = 0.0
            
            uncertainties[i] = max(0.0, min(1.0, uncertainty))
        
        return uncertainties

    def interactive_clustering(self, embeddings: np.ndarray, 
                             user_constraints: Dict[str, Any]) -> np.ndarray:
        """
        Perform clustering with user-provided constraints.
        """
        # Extract constraints
        must_link = user_constraints.get('must_link', [])  # Pairs that must be in same cluster
        cannot_link = user_constraints.get('cannot_link', [])  # Pairs that cannot be in same cluster
        seed_clusters = user_constraints.get('seed_clusters', {})  # Pre-assigned points
        
        # Start with unconstrained clustering
        base_algorithm = user_constraints.get('base_algorithm', 'kmeans')
        num_clusters = user_constraints.get('num_clusters', 4)
        
        labels = self.cluster(embeddings, base_algorithm, num_clusters)
        
        # Apply constraints iteratively
        labels = self._apply_must_link_constraints(embeddings, labels, must_link)
        labels = self._apply_cannot_link_constraints(embeddings, labels, cannot_link)
        labels = self._apply_seed_constraints(labels, seed_clusters)
        
        return labels

    def _apply_must_link_constraints(self, embeddings: np.ndarray, 
                                   labels: np.ndarray, 
                                   must_link: List[Tuple[int, int]]) -> np.ndarray:
        """
        Apply must-link constraints by merging clusters.
        """
        constrained_labels = labels.copy()
        
        for point1, point2 in must_link:
            if point1 < len(labels) and point2 < len(labels):
                label1, label2 = constrained_labels[point1], constrained_labels[point2]
                if label1 != label2:
                    # Merge clusters by assigning all points of label2 to label1
                    constrained_labels[constrained_labels == label2] = label1
        
        return self._relabel_consecutive(constrained_labels)

    def _apply_cannot_link_constraints(self, embeddings: np.ndarray, 
                                     labels: np.ndarray, 
                                     cannot_link: List[Tuple[int, int]]) -> np.ndarray:
        """
        Apply cannot-link constraints by splitting clusters if necessary.
        """
        constrained_labels = labels.copy()
        
        for point1, point2 in cannot_link:
            if point1 < len(labels) and point2 < len(labels):
                if constrained_labels[point1] == constrained_labels[point2]:
                    # Split cluster - assign point2 to a new cluster
                    new_cluster_id = max(constrained_labels) + 1
                    constrained_labels[point2] = new_cluster_id
        
        return self._relabel_consecutive(constrained_labels)

    def _apply_seed_constraints(self, labels: np.ndarray, 
                              seed_clusters: Dict[int, int]) -> np.ndarray:
        """
        Apply seed cluster constraints by fixing certain point assignments.
        """
        constrained_labels = labels.copy()
        
        for point_idx, cluster_id in seed_clusters.items():
            if point_idx < len(labels):
                constrained_labels[point_idx] = cluster_id
        
        return self._relabel_consecutive(constrained_labels)

    def ensemble_clustering(self, embeddings: np.ndarray, 
                          algorithms: List[str] = None,
                          num_clusters: int = 4) -> np.ndarray:
        """
        Perform ensemble clustering by combining multiple algorithms.
        """
        if algorithms is None:
            algorithms = ['kmeans', 'hdbscan', 'gaussian_mixture']
        
        # Run multiple clustering algorithms
        all_labels = []
        for algorithm in algorithms:
            try:
                labels = self.cluster(embeddings, algorithm, num_clusters)
                all_labels.append(labels)
            except Exception as e:
                logger.warning(f"Error in {algorithm}: {str(e)}")
        
        if not all_labels:
            # Fallback to kmeans
            return self.cluster(embeddings, 'kmeans', num_clusters)
        
        # Combine results using consensus clustering
        consensus_labels = self._consensus_clustering(all_labels)
        
        return consensus_labels

    def _consensus_clustering(self, all_labels: List[np.ndarray]) -> np.ndarray:
        """
        Create consensus clustering from multiple clustering results.
        """
        n_points = len(all_labels[0])
        n_clusterings = len(all_labels)
        
        # Create co-association matrix
        co_matrix = np.zeros((n_points, n_points))
        
        for labels in all_labels:
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if labels[i] == labels[j]:
                        co_matrix[i, j] += 1
                        co_matrix[j, i] += 1
        
        # Normalize co-association matrix
        co_matrix /= n_clusterings
        
        # Convert similarity matrix to distance matrix
        distance_matrix = 1.0 - co_matrix
        
        # Apply hierarchical clustering on distance matrix
        from sklearn.cluster import AgglomerativeClustering
        
        # Estimate number of clusters from the most common result
        cluster_counts = [len(np.unique(labels)) for labels in all_labels]
        consensus_k = int(np.median(cluster_counts))
        
        clustering = AgglomerativeClustering(
            n_clusters=consensus_k,
            metric='precomputed',
            linkage='average'
        )
        
        consensus_labels = clustering.fit_predict(distance_matrix)
        
        return consensus_labels