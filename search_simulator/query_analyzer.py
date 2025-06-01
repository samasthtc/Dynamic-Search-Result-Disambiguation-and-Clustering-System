"""
Query analysis module
Analyzes query characteristics and provides clustering recommendations
"""

import random
from typing import Dict, List, Any
import logging

from .data_templates import AMBIGUOUS_QUERIES

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes search queries to determine their characteristics
    and provide optimal clustering recommendations.
    """
    
    def __init__(self):
        self.complexity_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Perform comprehensive query analysis.
        
        Args:
            query: Search query to analyze
            
        Returns:
            Complete analysis with recommendations
        """
        query_lower = query.lower().strip()
        
        # Get basic characteristics
        if query_lower in AMBIGUOUS_QUERIES:
            basic_info = AMBIGUOUS_QUERIES[query_lower].copy()
        else:
            basic_info = self._analyze_unknown_query(query)
        
        # Perform detailed analysis
        complexity = self._assess_complexity(basic_info, query)
        recommendations = self._generate_clustering_recommendations(complexity, basic_info)
        search_characteristics = self._analyze_search_characteristics(query)
        
        return {
            'query': query,
            'basic_characteristics': basic_info,
            'complexity_assessment': complexity,
            'clustering_recommendations': recommendations,
            'search_characteristics': search_characteristics,
            'optimization_suggestions': self._generate_optimization_suggestions(complexity)
        }
    
    def _analyze_unknown_query(self, query: str) -> Dict[str, Any]:
        """Analyze characteristics of unknown queries"""
        words = query.lower().split()
        word_count = len(words)
        
        # Estimate ambiguity based on query structure
        if word_count == 1:
            ambiguity_level = 0.7  # Single words are often ambiguous
            difficulty = 'high'
        elif word_count == 2:
            ambiguity_level = 0.5
            difficulty = 'medium'
        else:
            ambiguity_level = 0.3  # Longer queries are usually more specific
            difficulty = 'low'
        
        # Detect potential entity types
        entity_types = self._detect_entity_types(query)
        
        return {
            'ambiguity_level': ambiguity_level,
            'entity_types': entity_types,
            'difficulty': difficulty,
            'search_volume': self._estimate_search_volume(query),
            'competition': self._estimate_competition(query),
            'seasonal_variance': random.uniform(0.1, 0.3)
        }
    
    def _detect_entity_types(self, query: str) -> List[str]:
        """Detect potential entity types in query"""
        query_lower = query.lower()
        entity_types = []
        
        # Simple heuristics for entity detection
        if any(indicator in query_lower for indicator in ['inc', 'corp', 'company', 'ltd']):
            entity_types.append('organization')
        
        if any(indicator in query_lower for indicator in ['mr', 'dr', 'prof', 'john', 'mary']):
            entity_types.append('person')
        
        if any(indicator in query_lower for indicator in ['city', 'country', 'street', 'avenue']):
            entity_types.append('location')
        
        if any(indicator in query_lower for indicator in ['programming', 'software', 'app', 'tech']):
            entity_types.append('technology')
        
        if any(indicator in query_lower for indicator in ['buy', 'price', 'cost', 'product']):
            entity_types.append('product')
        
        return entity_types if entity_types else ['general']
    
    def _assess_complexity(self, basic_info: Dict, query: str) -> Dict[str, Any]:
        """Assess overall query complexity for clustering"""
        complexity_factors = []
        complexity_score = 0.0
        
        # Ambiguity level contribution (40% weight)
        ambiguity = basic_info.get('ambiguity_level', 0.5)
        complexity_score += ambiguity * 0.4
        
        if ambiguity > 0.8:
            complexity_factors.append('very_high_ambiguity')
        elif ambiguity > 0.6:
            complexity_factors.append('high_ambiguity')
        
        # Entity type diversity (20% weight)
        entity_types = basic_info.get('entity_types', [])
        entity_diversity = len(entity_types) / 5.0  # Normalize to max 5 types
        complexity_score += entity_diversity * 0.2
        
        if len(entity_types) > 3:
            complexity_factors.append('multiple_entity_types')
        
        # Query length factor (15% weight)
        word_count = len(query.split())
        if word_count == 1:
            length_factor = 0.8  # Single words are complex
            complexity_factors.append('single_word_query')
        elif word_count > 5:
            length_factor = 0.3  # Very long queries are less ambiguous
            complexity_factors.append('long_query')
        else:
            length_factor = 0.5
        
        complexity_score += length_factor * 0.15
        
        # Seasonal variance (10% weight)
        seasonal_var = basic_info.get('seasonal_variance', 0.1)
        complexity_score += seasonal_var * 0.1
        
        if seasonal_var > 0.3:
            complexity_factors.append('seasonal_sensitive')
        
        # Competition level (10% weight)
        competition = basic_info.get('competition', 'medium')
        competition_score = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'very_high': 1.0}
        complexity_score += competition_score.get(competition, 0.5) * 0.1
        
        # Search volume impact (5% weight)
        search_volume = basic_info.get('search_volume', 'medium')
        volume_score = {'low': 0.2, 'medium': 0.5, 'high': 0.7, 'very_high': 1.0}
        complexity_score += volume_score.get(search_volume, 0.5) * 0.05
        
        # Determine complexity level
        if complexity_score > self.complexity_thresholds['high']:
            complexity_level = 'very_high'
        elif complexity_score > self.complexity_thresholds['medium']:
            complexity_level = 'high'
        elif complexity_score > self.complexity_thresholds['low']:
            complexity_level = 'medium'
        else:
            complexity_level = 'low'
        
        return {
            'complexity_score': complexity_score,
            'complexity_level': complexity_level,
            'complexity_factors': complexity_factors,
            'clustering_difficulty': complexity_level,
            'recommended_min_cluster_size': self._recommend_min_cluster_size(complexity_level),
            'recommended_max_clusters': self._recommend_max_clusters(complexity_score)
        }
    
    def _recommend_min_cluster_size(self, complexity_level: str) -> int:
        """Recommend minimum cluster size based on complexity"""
        recommendations = {
            'low': 2,
            'medium': 2,
            'high': 3,
            'very_high': 3
        }
        return recommendations.get(complexity_level, 2)
    
    def _recommend_max_clusters(self, complexity_score: float) -> int:
        """Recommend maximum number of clusters"""
        if complexity_score > 0.8:
            return 8
        elif complexity_score > 0.6:
            return 6
        elif complexity_score > 0.4:
            return 5
        else:
            return 4
    
    def _generate_clustering_recommendations(self, complexity: Dict, 
                                           basic_info: Dict) -> Dict[str, Any]:
        """Generate clustering algorithm and parameter recommendations"""
        complexity_level = complexity['complexity_level']
        ambiguity = basic_info.get('ambiguity_level', 0.5)
        
        recommendations = {
            'primary_algorithm': 'kmeans',
            'alternative_algorithms': [],
            'parameters': {},
            'preprocessing_steps': [],
            'evaluation_metrics': ['silhouette_score', 'cluster_purity']
        }
        
        # Algorithm selection based on complexity
        if complexity_level == 'very_high':
            recommendations['primary_algorithm'] = 'ensemble'
            recommendations['alternative_algorithms'] = ['adaptive', 'hdbscan', 'gaussian_mixture']
            recommendations['parameters'] = {
                'ensemble_methods': ['kmeans', 'hdbscan', 'gaussian_mixture'],
                'voting_strategy': 'consensus',
                'min_cluster_size': complexity['recommended_min_cluster_size']
            }
            recommendations['preprocessing_steps'].extend([
                'semantic_embedding_enhancement',
                'query_expansion',
                'context_enrichment'
            ])
            recommendations['evaluation_metrics'].extend([
                'normalized_mutual_info',
                'user_satisfaction_score'
            ])
            
        elif complexity_level == 'high':
            recommendations['primary_algorithm'] = 'adaptive'
            recommendations['alternative_algorithms'] = ['hdbscan', 'gaussian_mixture', 'ensemble']
            recommendations['parameters'] = {
                'auto_tune_parameters': True,
                'min_cluster_size': complexity['recommended_min_cluster_size'],
                'algorithm_selection_criteria': ['silhouette', 'stability']
            }
            recommendations['preprocessing_steps'].extend([
                'semantic_embedding_enhancement',
                'outlier_detection'
            ])
            recommendations['evaluation_metrics'].append('adjusted_rand_index')
            
        elif complexity_level == 'medium':
            recommendations['primary_algorithm'] = 'gaussian_mixture'
            recommendations['alternative_algorithms'] = ['kmeans', 'hdbscan', 'adaptive']
            recommendations['parameters'] = {
                'n_components': 'auto_select',
                'covariance_type': 'full',
                'init_params': 'kmeans'
            }
            recommendations['preprocessing_steps'].append('feature_scaling')
            
        else:  # low complexity
            recommendations['primary_algorithm'] = 'kmeans'
            recommendations['alternative_algorithms'] = ['hierarchical', 'gaussian_mixture']
            recommendations['parameters'] = {
                'n_clusters': 4,
                'init': 'k-means++',
                'n_init': 10
            }
        
        # Additional recommendations based on specific factors
        if 'seasonal_sensitive' in complexity['complexity_factors']:
            recommendations['preprocessing_steps'].append('temporal_weighting')
        
        if 'multiple_entity_types' in complexity['complexity_factors']:
            recommendations['preprocessing_steps'].append('entity_aware_features')
        
        if ambiguity > 0.8:
            recommendations['preprocessing_steps'].append('disambiguation_context')
        
        # Parameter refinements
        recommendations['parameters'].update({
            'max_clusters': complexity['recommended_max_clusters'],
            'distance_metric': 'euclidean' if complexity_level in ['low', 'medium'] else 'cosine',
            'random_state': 42,
            'convergence_tolerance': 1e-4
        })
        
        return recommendations
    
    def _analyze_search_characteristics(self, query: str) -> Dict[str, Any]:
        """Analyze search-specific characteristics"""
        return {
            'estimated_search_volume': self._estimate_search_volume(query),
            'competition_level': self._estimate_competition(query),
            'commercial_intent': self._detect_commercial_intent(query),
            'local_intent': self._detect_local_intent(query),
            'informational_intent': self._detect_informational_intent(query),
            'query_length_category': self._categorize_query_length(query),
            'specificity_level': self._assess_query_specificity(query)
        }
    
    def _estimate_search_volume(self, query: str) -> str:
        """Estimate relative search volume"""
        query_lower = query.lower()
        
        # High volume terms (simplified heuristic)
        high_volume_terms = [
            'apple', 'google', 'facebook', 'amazon', 'microsoft',
            'python', 'java', 'javascript', 'covid', 'weather',
            'news', 'email', 'youtube', 'netflix'
        ]
        
        if any(term in query_lower for term in high_volume_terms):
            return 'very_high'
        
        # Estimate based on query characteristics
        word_count = len(query.split())
        if word_count == 1:
            return 'high'
        elif word_count == 2:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_competition(self, query: str) -> str:
        """Estimate SEO competition level"""
        word_count = len(query.split())
        
        # Longer queries typically have lower competition
        if word_count == 1:
            return random.choice(['high', 'very_high'])
        elif word_count == 2:
            return random.choice(['medium', 'high'])
        elif word_count <= 4:
            return random.choice(['low', 'medium'])
        else:
            return 'low'
    
    def _detect_commercial_intent(self, query: str) -> float:
        """Detect commercial intent (0.0 to 1.0)"""
        commercial_keywords = [
            'buy', 'purchase', 'price', 'cost', 'deal', 'sale', 'shop',
            'store', 'order', 'cheap', 'discount', 'review', 'compare',
            'best', 'top', 'vs', 'versus'
        ]
        
        query_lower = query.lower()
        matches = sum(1 for keyword in commercial_keywords if keyword in query_lower)
        
        return min(1.0, matches / 3.0)  # Normalize to max 3 signals
    
    def _detect_local_intent(self, query: str) -> float:
        """Detect local/geographic intent (0.0 to 1.0)"""
        local_keywords = [
            'near', 'nearby', 'local', 'in', 'at', 'location', 'address',
            'hours', 'open', 'closed', 'directions', 'map', 'around'
        ]
        
        query_lower = query.lower()
        matches = sum(1 for keyword in local_keywords if keyword in query_lower)
        
        return min(1.0, matches / 2.0)  # Normalize to max 2 signals
    
    def _detect_informational_intent(self, query: str) -> float:
        """Detect informational intent (0.0 to 1.0)"""
        informational_keywords = [
            'what', 'how', 'why', 'when', 'where', 'who', 'guide',
            'tutorial', 'learn', 'explain', 'definition', 'meaning',
            'examples', 'tips', 'help'
        ]
        
        query_lower = query.lower()
        matches = sum(1 for keyword in informational_keywords if keyword in query_lower)
        
        return min(1.0, matches / 2.0)  # Normalize to max 2 signals
    
    def _categorize_query_length(self, query: str) -> str:
        """Categorize query by length"""
        word_count = len(query.split())
        
        if word_count == 1:
            return 'single_word'
        elif word_count == 2:
            return 'short'
        elif word_count <= 4:
            return 'medium'
        elif word_count <= 7:
            return 'long'
        else:
            return 'very_long'
    
    def _assess_query_specificity(self, query: str) -> str:
        """Assess how specific the query is"""
        word_count = len(query.split())
        
        # Check for specific indicators
        specific_indicators = [
            'model', 'version', 'year', 'brand', 'specific', 'exact',
            'particular', 'detailed', '2023', '2024'
        ]
        
        query_lower = query.lower()
        has_specific_terms = any(indicator in query_lower for indicator in specific_indicators)
        
        if word_count >= 4 or has_specific_terms:
            return 'high'
        elif word_count == 3:
            return 'medium'
        elif word_count == 2:
            return 'low'
        else:
            return 'very_low'
    
    def _generate_optimization_suggestions(self, complexity: Dict) -> List[str]:
        """Generate optimization suggestions based on complexity analysis"""
        suggestions = []
        complexity_level = complexity['complexity_level']
        factors = complexity['complexity_factors']
        
        # General suggestions based on complexity level
        if complexity_level == 'very_high':
            suggestions.extend([
                "Use ensemble clustering for robust results across different algorithms",
                "Implement context-aware embeddings for better semantic understanding",
                "Consider user feedback integration for iterative improvement",
                "Apply query expansion techniques to capture related concepts"
            ])
        elif complexity_level == 'high':
            suggestions.extend([
                "Use adaptive algorithm selection for optimal performance",
                "Implement semantic similarity weighting in clustering",
                "Consider hierarchical clustering for better interpretability"
            ])
        elif complexity_level == 'medium':
            suggestions.extend([
                "Gaussian Mixture Models work well for moderate complexity",
                "Apply feature scaling for improved clustering performance",
                "Consider cluster validation metrics for parameter tuning"
            ])
        else:
            suggestions.extend([
                "K-means clustering should provide good results",
                "Standard preprocessing steps are sufficient",
                "Focus on cluster interpretation rather than algorithmic complexity"
            ])
        
        # Factor-specific suggestions
        if 'very_high_ambiguity' in factors:
            suggestions.append("Implement disambiguation techniques using contextual clues")
        
        if 'multiple_entity_types' in factors:
            suggestions.append("Use entity-aware features to improve clustering accuracy")
        
        if 'seasonal_sensitive' in factors:
            suggestions.append("Apply temporal weighting to account for seasonal variations")
        
        if 'single_word_query' in factors:
            suggestions.append("Single-word queries benefit from semantic expansion")
        
        # Limit to top 5 suggestions
        return suggestions[:5]
    
    def get_clustering_difficulty_score(self, query: str) -> float:
        """Get a simple 0-1 difficulty score for clustering this query"""
        analysis = self.analyze(query)
        return analysis['complexity_assessment']['complexity_score']
    
    def recommend_algorithms_ranked(self, query: str) -> List[Dict[str, Any]]:
        """Get ranked list of recommended algorithms with scores"""
        analysis = self.analyze(query)
        complexity = analysis['complexity_assessment']['complexity_level']
        
        # Algorithm suitability scores based on complexity
        algorithm_scores = {
            'very_high': {
                'ensemble': 0.95,
                'adaptive': 0.90,
                'hdbscan': 0.85,
                'gaussian_mixture': 0.80,
                'kmeans': 0.60,
                'hierarchical': 0.70
            },
            'high': {
                'adaptive': 0.95,
                'hdbscan': 0.90,
                'gaussian_mixture': 0.85,
                'ensemble': 0.88,
                'kmeans': 0.70,
                'hierarchical': 0.75
            },
            'medium': {
                'gaussian_mixture': 0.95,
                'kmeans': 0.90,
                'hdbscan': 0.85,
                'adaptive': 0.80,
                'hierarchical': 0.75,
                'ensemble': 0.70
            },
            'low': {
                'kmeans': 0.95,
                'hierarchical': 0.90,
                'gaussian_mixture': 0.85,
                'hdbscan': 0.70,
                'adaptive': 0.65,
                'ensemble': 0.60
            }
        }
        
        scores = algorithm_scores.get(complexity, algorithm_scores['medium'])
        
        # Sort by score and create ranked list
        ranked_algorithms = []
        for algorithm, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            ranked_algorithms.append({
                'algorithm': algorithm,
                'suitability_score': score,
                'recommendation_reason': self._get_algorithm_reason(algorithm, complexity)
            })
        
        return ranked_algorithms
    
    def _get_algorithm_reason(self, algorithm: str, complexity: str) -> str:
        """Get reason for algorithm recommendation"""
        reasons = {
            'ensemble': f"Combines multiple algorithms for robust results with {complexity} complexity",
            'adaptive': f"Automatically selects best approach for {complexity} complexity queries",
            'hdbscan': f"Handles noise and varying cluster densities well for {complexity} queries",
            'gaussian_mixture': f"Good probabilistic clustering for {complexity} ambiguity levels",
            'kmeans': f"Simple and effective for {complexity} complexity scenarios",
            'hierarchical': f"Provides interpretable cluster hierarchy for {complexity} queries"
        }
        
        return reasons.get(algorithm, f"Suitable for {complexity} complexity queries")
