import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    calinski_harabasz_score, davies_bouldin_score, homogeneity_score,
    completeness_score, v_measure_score
)
from collections import Counter, defaultdict
import logging
from typing import List, Dict, Any, Tuple, Optional
import math

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Comprehensive evaluation metrics calculator for clustering quality assessment.
    Implements both traditional clustering metrics and search-specific measures.
    """
    
    def __init__(self):
        self.metric_history = defaultdict(list)
        self.baseline_scores = {
            'cluster_purity': 0.5,
            'adjusted_rand_index': 0.0,
            'silhouette_score': 0.0,
            'normalized_mutual_info': 0.0
        }
        
    def calculate_all_metrics(self, search_results: List[Dict], clusters: List[Dict], 
                            cluster_labels: List[int], true_labels: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive clustering evaluation metrics.
        
        Args:
            search_results: Original search results
            clusters: Clustered results
            cluster_labels: Predicted cluster labels
            true_labels: Ground truth labels (categories)
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        try:
            # Extract embeddings if available
            embeddings = np.array([result.get('embedding', []) for result in search_results])
            if embeddings.size == 0 or len(embeddings[0]) == 0:
                embeddings = self._generate_fallback_embeddings(search_results)
            
            # Internal clustering metrics (no ground truth needed)
            if len(set(cluster_labels)) > 1 and len(embeddings) > 1:
                metrics.update(self._calculate_internal_metrics(embeddings, cluster_labels))
            
            # External clustering metrics (require ground truth)
            if len(set(true_labels)) > 1:
                metrics.update(self._calculate_external_metrics(cluster_labels, true_labels))
            
            # Search-specific metrics
            metrics.update(self._calculate_search_metrics(search_results, clusters))
            
            # Cluster quality metrics
            metrics.update(self._calculate_cluster_quality_metrics(clusters, search_results))
            
            # Store metrics in history
            for metric_name, value in metrics.items():
                self.metric_history[metric_name].append(value)
            
            logger.info(f"Calculated {len(metrics)} evaluation metrics")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            metrics = self._get_default_metrics()
        
        return metrics

    def _calculate_internal_metrics(self, embeddings: np.ndarray, 
                                  cluster_labels: np.ndarray) -> Dict[str, float]:
        """Calculate internal clustering validation metrics"""
        metrics = {}
        
        try:
            # Silhouette Score
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                metrics['silhouette_score'] = max(0.0, silhouette_avg)
            else:
                metrics['silhouette_score'] = 0.0
            
            # Calinski-Harabasz Score (Variance Ratio Criterion)
            if len(set(cluster_labels)) > 1:
                ch_score = calinski_harabasz_score(embeddings, cluster_labels)
                # Normalize to 0-1 range (rough approximation)
                metrics['calinski_harabasz_score'] = min(1.0, ch_score / 1000)
            else:
                metrics['calinski_harabasz_score'] = 0.0
            
            # Davies-Bouldin Score (lower is better, so we invert)
            if len(set(cluster_labels)) > 1:
                db_score = davies_bouldin_score(embeddings, cluster_labels)
                metrics['davies_bouldin_score'] = max(0.0, 1.0 / (1.0 + db_score))
            else:
                metrics['davies_bouldin_score'] = 0.0
            
        except Exception as e:
            logger.warning(f"Error in internal metrics calculation: {str(e)}")
            metrics = {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': 0.0
            }
        
        return metrics

    def _calculate_external_metrics(self, cluster_labels: List[int], 
                                  true_labels: List[str]) -> Dict[str, float]:
        """Calculate external clustering validation metrics"""
        metrics = {}
        
        try:
            # Convert string labels to integers for consistency
            true_labels_int = self._encode_string_labels(true_labels)
            
            # Adjusted Rand Index
            ari = adjusted_rand_score(true_labels_int, cluster_labels)
            metrics['adjusted_rand_index'] = max(0.0, ari)
            
            # Normalized Mutual Information
            nmi = normalized_mutual_info_score(true_labels_int, cluster_labels)
            metrics['normalized_mutual_info'] = nmi
            
            # Homogeneity, Completeness, and V-measure
            homogeneity = homogeneity_score(true_labels_int, cluster_labels)
            completeness = completeness_score(true_labels_int, cluster_labels)
            v_measure = v_measure_score(true_labels_int, cluster_labels)
            
            metrics['homogeneity_score'] = homogeneity
            metrics['completeness_score'] = completeness
            metrics['v_measure_score'] = v_measure
            
            # Cluster Purity
            purity = self._calculate_purity(cluster_labels, true_labels_int)
            metrics['cluster_purity'] = purity
            
        except Exception as e:
            logger.warning(f"Error in external metrics calculation: {str(e)}")
            metrics = {
                'adjusted_rand_index': 0.0,
                'normalized_mutual_info': 0.0,
                'homogeneity_score': 0.0,
                'completeness_score': 0.0,
                'v_measure_score': 0.0,
                'cluster_purity': 0.0
            }
        
        return metrics

    def _calculate_search_metrics(self, search_results: List[Dict], 
                                clusters: List[Dict]) -> Dict[str, float]:
        """Calculate search-specific quality metrics"""
        metrics = {}
        
        try:
            # Result Coverage - how many results are clustered
            total_results = len(search_results)
            clustered_results = sum(len(cluster['results']) for cluster in clusters)
            metrics['result_coverage'] = clustered_results / total_results if total_results > 0 else 0.0
            
            # Cluster Balance - measure of cluster size distribution
            if clusters:
                cluster_sizes = [cluster['size'] for cluster in clusters]
                size_std = np.std(cluster_sizes)
                size_mean = np.mean(cluster_sizes)
                balance_score = 1.0 - (size_std / size_mean) if size_mean > 0 else 0.0
                metrics['cluster_balance'] = max(0.0, balance_score)
            else:
                metrics['cluster_balance'] = 0.0
            
            # Topic Coherence - semantic coherence within clusters
            coherence_scores = []
            for cluster in clusters:
                if 'coherence_score' in cluster:
                    coherence_scores.append(cluster['coherence_score'])
                else:
                    # Calculate coherence based on result similarity
                    coherence = self._calculate_topic_coherence(cluster['results'])
                    coherence_scores.append(coherence)
            
            metrics['avg_topic_coherence'] = np.mean(coherence_scores) if coherence_scores else 0.0
            
            # Diversity Score - how diverse are the clusters
            diversity_scores = []
            for cluster in clusters:
                if 'diversity_score' in cluster:
                    diversity_scores.append(cluster['diversity_score'])
                else:
                    diversity = self._calculate_cluster_diversity(cluster['results'])
                    diversity_scores.append(diversity)
            
            metrics['avg_cluster_diversity'] = np.mean(diversity_scores) if diversity_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error in search metrics calculation: {str(e)}")
            metrics = {
                'result_coverage': 1.0,
                'cluster_balance': 0.5,
                'avg_topic_coherence': 0.5,
                'avg_cluster_diversity': 0.5
            }
        
        return metrics

    def _calculate_cluster_quality_metrics(self, clusters: List[Dict], 
                                         search_results: List[Dict]) -> Dict[str, float]:
        """Calculate advanced cluster quality metrics"""
        metrics = {}
        
        try:
            if not clusters:
                return {'cluster_quality_score': 0.0}
            
            # Inter-cluster separation
            separation_scores = []
            for i, cluster1 in enumerate(clusters):
                for j, cluster2 in enumerate(clusters[i+1:], i+1):
                    separation = self._calculate_cluster_separation(cluster1, cluster2)
                    separation_scores.append(separation)
            
            metrics['avg_cluster_separation'] = np.mean(separation_scores) if separation_scores else 0.0
            
            # Intra-cluster cohesion
            cohesion_scores = []
            for cluster in clusters:
                cohesion = self._calculate_cluster_cohesion(cluster)
                cohesion_scores.append(cohesion)
            
            metrics['avg_cluster_cohesion'] = np.mean(cohesion_scores) if cohesion_scores else 0.0
            
            # Overall cluster quality score
            quality_components = [
                metrics.get('avg_cluster_cohesion', 0.0),
                metrics.get('avg_cluster_separation', 0.0),
                metrics.get('cluster_balance', 0.0),
                metrics.get('avg_topic_coherence', 0.0)
            ]
            metrics['cluster_quality_score'] = np.mean(quality_components)
            
            # Relevance-based clustering score
            relevance_score = self._calculate_relevance_based_score(clusters, search_results)
            metrics['relevance_clustering_score'] = relevance_score
            
        except Exception as e:
            logger.warning(f"Error in cluster quality metrics: {str(e)}")
            metrics = {'cluster_quality_score': 0.5}
        
        return metrics

    def _calculate_purity(self, cluster_labels: List[int], true_labels: List[int]) -> float:
        """Calculate cluster purity score"""
        if len(cluster_labels) == 0:
            return 0.0
        
        # Group items by cluster
        clusters = defaultdict(list)
        for i, cluster_id in enumerate(cluster_labels):
            clusters[cluster_id].append(true_labels[i])
        
        total_purity = 0
        total_items = len(cluster_labels)
        
        for cluster_id, items in clusters.items():
            # Find most common true label in this cluster
            label_counts = Counter(items)
            max_count = max(label_counts.values()) if label_counts else 0
            total_purity += max_count
        
        return total_purity / total_items if total_items > 0 else 0.0

    def _calculate_topic_coherence(self, results: List[Dict]) -> float:
        """Calculate topic coherence within a cluster based on text similarity"""
        if len(results) < 2:
            return 1.0
        
        # Calculate coherence based on title and snippet similarity
        texts = []
        for result in results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            texts.append(text.lower())
        
        # Simple coherence based on common words
        word_sets = [set(text.split()) for text in texts]
        
        coherence_scores = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i].intersection(word_sets[j]))
                union = len(word_sets[i].union(word_sets[j]))
                jaccard_sim = intersection / union if union > 0 else 0.0
                coherence_scores.append(jaccard_sim)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _calculate_cluster_diversity(self, results: List[Dict]) -> float:
        """Calculate diversity within a cluster"""
        if len(results) <= 1:
            return 0.0
        
        # Diversity based on different categories/domains
        categories = [result.get('category', 'unknown') for result in results]
        domains = [result.get('domain', 'unknown') for result in results]
        
        unique_categories = len(set(categories))
        unique_domains = len(set(domains))
        
        category_diversity = unique_categories / len(results)
        domain_diversity = unique_domains / len(results)
        
        return (category_diversity + domain_diversity) / 2

    def _calculate_cluster_separation(self, cluster1: Dict, cluster2: Dict) -> float:
        """Calculate separation between two clusters"""
        try:
            # Get embeddings for both clusters
            embeddings1 = [result.get('embedding', []) for result in cluster1['results']]
            embeddings2 = [result.get('embedding', []) for result in cluster2['results']]
            
            # Filter out empty embeddings
            embeddings1 = [emb for emb in embeddings1 if emb]
            embeddings2 = [emb for emb in embeddings2 if emb]
            
            if not embeddings1 or not embeddings2:
                return 0.5  # Default separation
            
            # Calculate centroids
            centroid1 = np.mean(embeddings1, axis=0)
            centroid2 = np.mean(embeddings2, axis=0)
            
            # Euclidean distance between centroids
            distance = np.linalg.norm(centroid1 - centroid2)
            
            # Normalize to 0-1 range (approximate)
            normalized_distance = min(1.0, distance / 10.0)
            
            return normalized_distance
            
        except Exception as e:
            logger.warning(f"Error calculating cluster separation: {str(e)}")
            return 0.5

    def _calculate_cluster_cohesion(self, cluster: Dict) -> float:
        """Calculate cohesion within a cluster"""
        try:
            results = cluster['results']
            if len(results) < 2:
                return 1.0
            
            # Use precomputed coherence score if available
            if 'coherence_score' in cluster:
                return cluster['coherence_score']
            
            # Calculate based on embedding similarity
            embeddings = [result.get('embedding', []) for result in results]
            embeddings = [emb for emb in embeddings if emb]
            
            if len(embeddings) < 2:
                return 0.5
            
            # Calculate average pairwise similarity
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating cluster cohesion: {str(e)}")
            return 0.5

    def _calculate_relevance_based_score(self, clusters: List[Dict], 
                                       search_results: List[Dict]) -> float:
        """Calculate clustering score based on relevance preservation"""
        try:
            if not clusters or not search_results:
                return 0.0
            
            # Calculate how well clustering preserves relevance ordering
            relevance_scores = [result.get('relevance_score', 0.5) for result in search_results]
            
            cluster_relevance_scores = []
            for cluster in clusters:
                cluster_relevances = [result.get('relevance_score', 0.5) 
                                    for result in cluster['results']]
                avg_relevance = np.mean(cluster_relevances) if cluster_relevances else 0.0
                cluster_relevance_scores.append(avg_relevance)
            
            # Calculate variance in cluster relevance scores
            # Lower variance means better relevance preservation
            relevance_variance = np.var(cluster_relevance_scores) if cluster_relevance_scores else 1.0
            relevance_preservation = 1.0 / (1.0 + relevance_variance)
            
            return relevance_preservation
            
        except Exception as e:
            logger.warning(f"Error calculating relevance-based score: {str(e)}")
            return 0.5

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            return 0.0

    def _encode_string_labels(self, labels: List[str]) -> List[int]:
        """Convert string labels to integer labels"""
        unique_labels = list(set(labels))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        return [label_to_int[label] for label in labels]

    def _generate_fallback_embeddings(self, search_results: List[Dict]) -> np.ndarray:
        """Generate simple embeddings based on text features"""
        embeddings = []
        
        for result in search_results:
            # Simple feature-based embedding
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            text = f"{title} {snippet}".lower()
            
            # Basic features
            features = [
                len(text),  # Text length
                len(text.split()),  # Word count
                text.count('the'),  # Article frequency
                text.count('and'),  # Conjunction frequency
                len(set(text.split())),  # Unique words
                result.get('relevance_score', 0.5),  # Relevance
                hash(result.get('category', '')) % 100 / 100,  # Category hash
                hash(result.get('domain', '')) % 100 / 100,  # Domain hash
            ]
            
            embeddings.append(features)
        
        return np.array(embeddings)

    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when calculation fails"""
        return {
            'cluster_purity': 0.5,
            'adjusted_rand_index': 0.0,
            'silhouette_score': 0.0,
            'normalized_mutual_info': 0.0,
            'homogeneity_score': 0.0,
            'completeness_score': 0.0,
            'v_measure_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_score': 0.0,
            'result_coverage': 1.0,
            'cluster_balance': 0.5,
            'avg_topic_coherence': 0.5,
            'avg_cluster_diversity': 0.5,
            'avg_cluster_separation': 0.5,
            'avg_cluster_cohesion': 0.5,
            'cluster_quality_score': 0.5,
            'relevance_clustering_score': 0.5
        }

    def get_metric_trends(self, window_size: int = 10) -> Dict[str, Dict[str, float]]:
        """Calculate trends for all metrics over recent history"""
        trends = {}
        
        for metric_name, values in self.metric_history.items():
            if len(values) >= 2:
                recent_values = values[-window_size:]
                
                # Calculate trend
                if len(recent_values) >= 2:
                    x = np.arange(len(recent_values))
                    trend_slope = np.polyfit(x, recent_values, 1)[0]
                else:
                    trend_slope = 0.0
                
                trends[metric_name] = {
                    'current_value': values[-1],
                    'average_value': np.mean(recent_values),
                    'trend_slope': trend_slope,
                    'improvement': trend_slope > 0,
                    'stability': np.std(recent_values) if len(recent_values) > 1 else 0.0
                }
        
        return trends

    def generate_quality_report(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate a comprehensive quality assessment report"""
        report = {
            'overall_score': 0.0,
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'metric_grades': {}
        }
        
        # Define metric weights and thresholds
        metric_config = {
            'cluster_purity': {'weight': 0.2, 'excellent': 0.8, 'good': 0.6, 'poor': 0.4},
            'adjusted_rand_index': {'weight': 0.15, 'excellent': 0.7, 'good': 0.5, 'poor': 0.3},
            'silhouette_score': {'weight': 0.15, 'excellent': 0.6, 'good': 0.4, 'poor': 0.2},
            'avg_topic_coherence': {'weight': 0.15, 'excellent': 0.7, 'good': 0.5, 'poor': 0.3},
            'cluster_balance': {'weight': 0.1, 'excellent': 0.8, 'good': 0.6, 'poor': 0.4},
            'avg_cluster_separation': {'weight': 0.1, 'excellent': 0.7, 'good': 0.5, 'poor': 0.3},
            'relevance_clustering_score': {'weight': 0.15, 'excellent': 0.8, 'good': 0.6, 'poor': 0.4}
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, config in metric_config.items():
            value = current_metrics.get(metric_name, 0.0)
            weight = config['weight']
            
            # Assign grade
            if value >= config['excellent']:
                grade = 'A'
                grade_points = 4.0
            elif value >= config['good']:
                grade = 'B'
                grade_points = 3.0
            elif value >= config['poor']:
                grade = 'C'
                grade_points = 2.0
            else:
                grade = 'D'
                grade_points = 1.0
            
            report['metric_grades'][metric_name] = {
                'value': value,
                'grade': grade,
                'grade_points': grade_points
            }
            
            # Add to weighted score
            total_weighted_score += grade_points * weight
            total_weight += weight
            
            # Identify strengths and weaknesses
            if grade in ['A', 'B']:
                report['strengths'].append(f"Good {metric_name.replace('_', ' ')}")
            elif grade == 'D':
                report['weaknesses'].append(f"Poor {metric_name.replace('_', ' ')}")
        
        # Calculate overall score
        if total_weight > 0:
            report['overall_score'] = total_weighted_score / total_weight / 4.0  # Normalize to 0-1
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['metric_grades'])
        
        return report

    def _generate_recommendations(self, metric_grades: Dict[str, Dict]) -> List[str]:
        """Generate improvement recommendations based on metric grades"""
        recommendations = []
        
        for metric_name, grade_info in metric_grades.items():
            if grade_info['grade'] in ['C', 'D']:
                metric_readable = metric_name.replace('_', ' ').title()
                
                if 'purity' in metric_name:
                    recommendations.append(f"Improve cluster purity by refining clustering parameters")
                elif 'coherence' in metric_name:
                    recommendations.append(f"Enhance topic coherence by using semantic similarity")
                elif 'balance' in metric_name:
                    recommendations.append(f"Balance cluster sizes by adjusting minimum cluster size")
                elif 'separation' in metric_name:
                    recommendations.append(f"Increase cluster separation by using different distance metrics")
                else:
                    recommendations.append(f"Focus on improving {metric_readable}")
        
        if not recommendations:
            recommendations.append("Clustering quality is good across all metrics")
        
        return recommendations[:5]  # Limit to top 5 recommendations