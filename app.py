from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import hdbscan
import logging
import json
import pickle
from datetime import datetime
import threading
import time
from collections import defaultdict
import re
import requests
from typing import List, Dict, Any, Tuple
import arabic_reshaper
from bidi.algorithm import get_display
import os

# Custom imports for our components
from rl_agent import ReinforcementLearningAgent
from clustering_algorithms import ClusteringManager
from search_simulator.search_simulator import SearchSimulator
from evaluation_metrics import MetricsCalculator
from arabic_processor import ArabicTextProcessor

# Import JSON utilities
from json_utils import (
    NumpyEncoder, safe_json_serialize, clean_cluster_data, 
    clean_search_results, clean_metrics_data
)

app = Flask(__name__)
CORS(app)

# Configure JSON encoder for Flask
app.json_encoder = NumpyEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchDisambiguationSystem:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.arabic_processor = ArabicTextProcessor()
        self.clustering_manager = ClusteringManager()
        self.rl_agent = ReinforcementLearningAgent()
        self.search_simulator = SearchSimulator()
        self.metrics_calculator = MetricsCalculator()
        
        # System state
        self.current_results = []
        self.current_clusters = []
        self.user_feedback_history = []
        self.query_history = []
        self.system_metrics = {
            'total_queries': 0,
            'total_feedback_items': 0,
            'avg_user_satisfaction': 0.75,
            'rl_episodes': 0,
            'total_reward': 0.0
        }
        
        # Load pre-trained models if available
        self.load_pretrained_models()
        
        logger.info("Search Disambiguation System initialized successfully")

    def load_pretrained_models(self):
        """Load any pre-trained models or saved state with better error handling"""
        rl_agent_file = 'rl_agent_state.pkl'
        
        try:
            # Check if file exists and has content
            if os.path.exists(rl_agent_file) and os.path.getsize(rl_agent_file) > 0:
                with open(rl_agent_file, 'rb') as f:
                    loaded_agent = pickle.load(f)
                    
                # Validate the loaded agent
                if hasattr(loaded_agent, 'q_table') and hasattr(loaded_agent, 'state_size'):
                    self.rl_agent = loaded_agent
                    logger.info("Loaded pre-trained RL agent successfully")
                else:
                    logger.warning("Loaded RL agent appears corrupted, using fresh agent")
                    self._create_fresh_rl_agent()
            else:
                logger.info("No valid RL agent state file found, starting with fresh agent")
                self._create_fresh_rl_agent()
                
        except (EOFError, pickle.UnpicklingError, AttributeError) as e:
            logger.warning(f"Error loading RL agent state: {str(e)}")
            logger.info("Creating fresh RL agent and removing corrupted file")
            
            # Remove corrupted file
            if os.path.exists(rl_agent_file):
                try:
                    os.remove(rl_agent_file)
                    logger.info(f"Removed corrupted file: {rl_agent_file}")
                except OSError as remove_error:
                    logger.warning(f"Could not remove corrupted file: {remove_error}")
            
            self._create_fresh_rl_agent()
            
        except Exception as e:
            logger.error(f"Unexpected error loading RL agent: {str(e)}")
            self._create_fresh_rl_agent()

    def _create_fresh_rl_agent(self):
        """Create a fresh RL agent"""
        self.rl_agent = ReinforcementLearningAgent()
        logger.info("Created fresh RL agent")

    def save_system_state(self):
        """Save system state for persistence with better error handling"""
        try:
            # Save RL agent state
            rl_agent_file = 'rl_agent_state.pkl'
            with open(rl_agent_file, 'wb') as f:
                pickle.dump(self.rl_agent, f)
            logger.debug(f"Saved RL agent state to {rl_agent_file}")
            
            # Use safe serialization for metrics
            clean_metrics = clean_metrics_data(self.system_metrics)
            metrics_file = 'system_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(clean_metrics, f, indent=2)
            logger.debug(f"Saved system metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")

    def generate_embeddings(self, texts: List[str], language: str = 'en') -> np.ndarray:
        """Generate semantic embeddings for text content"""
        if language == 'ar':
            # Process Arabic text
            processed_texts = [self.arabic_processor.preprocess_text(text) for text in texts]
            embeddings = self.sentence_model.encode(processed_texts)
        else:
            embeddings = self.sentence_model.encode(texts)
        
        return embeddings

    def perform_search(self, query: str, language: str = 'en', num_results: int = 20) -> List[Dict]:
        """Simulate search results for a given query"""
        logger.info(f"Performing search for query: '{query}' in language: {language}")
        
        try:
            # Use search simulator to generate realistic results
            results = self.search_simulator.simulate_search(query, language, num_results)
            
            # The new simulator already includes embeddings and metadata,
            # but we need to ensure compatibility with the expected format
            enhanced_results = []
            for result in results:
                # Convert to expected format if needed
                enhanced_result = {
                    'id': result.get('id', len(enhanced_results)),
                    'title': result['title'],
                    'snippet': result['snippet'],
                    'url': result['url'],
                    'category': result.get('category', 'general'),
                    'domain': result.get('domain', 'unknown'),
                    'relevance_score': result.get('final_score', 0.5),
                    'embedding': self._generate_or_extract_embedding(result),
                    'language': result.get('language', language),
                    'authority_score': result.get('authority_score', 0.5),
                    'freshness_score': result.get('freshness_score', 0.5),
                    'social_signals': result.get('social_signals', {}),
                    'technical_score': result.get('technical_score', 0.5),
                    'publish_date': result.get('publish_date', '2023-01-01'),
                    'content_type': result.get('content_type', 'general')
                }
                enhanced_results.append(enhanced_result)
            
            # Clean results for JSON serialization
            self.current_results = clean_search_results(enhanced_results)
            
            self.query_history.append({
                'query': query,
                'language': language,
                'timestamp': datetime.now().isoformat(),
                'num_results': len(enhanced_results)
            })
            
            self.system_metrics['total_queries'] += 1
            return self.current_results
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            # Return empty results on error
            self.current_results = []
            return []

    def _generate_or_extract_embedding(self, result: Dict) -> List[float]:
        """Generate or extract embedding for a result"""
        try:
            # Check if embedding already exists
            if 'embedding' in result and result['embedding']:
                embedding = result['embedding']
                # Convert numpy array to list if needed
                if isinstance(embedding, np.ndarray):
                    return embedding.tolist()
                return embedding
            
            # Generate embedding from title and snippet
            text = f"{result['title']} {result['snippet']}"
            if result.get('language') == 'ar':
                # Process Arabic text
                processed_text = self.arabic_processor.preprocess_text(text)
                embedding = self.sentence_model.encode(processed_text)
            else:
                embedding = self.sentence_model.encode(text)
            
            # Convert to list for JSON serialization
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero embedding on error
            return [0.0] * 384  # Dimension of all-MiniLM-L6-v2

    def perform_clustering(self, algorithm: str = 'kmeans', num_clusters: int = 4, 
                          min_cluster_size: int = 2) -> List[Dict]:
        """Perform clustering on current search results"""
        if not self.current_results:
            logger.warning("No current results to cluster")
            return []
        
        logger.info(f"Performing clustering with algorithm: {algorithm}")
        
        try:
            # Extract embeddings
            embeddings = np.array([result['embedding'] for result in self.current_results])
            
            # Validate embeddings
            if embeddings.size == 0:
                logger.warning("No valid embeddings found for clustering")
                return []
            
            # Perform clustering
            cluster_labels = self.clustering_manager.cluster(
                embeddings, algorithm, num_clusters, min_cluster_size
            )
            
            # Organize results into clusters
            clusters = self._organize_clusters(cluster_labels)
            
            # Apply RL optimization
            optimized_clusters = self.rl_agent.optimize_clusters(
                clusters, self.user_feedback_history[-10:]  # Last 10 feedback items
            )
            
            # Clean clusters for JSON serialization
            self.current_clusters = clean_cluster_data(optimized_clusters)
            return self.current_clusters
            
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            # Return empty clusters on error
            return []

    def _organize_clusters(self, cluster_labels: np.ndarray) -> List[Dict]:
        """Organize search results into cluster structure"""
        cluster_dict = defaultdict(list)
        
        # Convert numpy types to Python types
        labels_list = [int(label) for label in cluster_labels]
        
        for i, label in enumerate(labels_list):
            if label != -1:  # -1 indicates noise/outlier in some algorithms
                cluster_dict[label].append(self.current_results[i])
        
        clusters = []
        for cluster_id, results in cluster_dict.items():
            if len(results) > 0:
                cluster = {
                    'id': int(cluster_id),  # Ensure Python int
                    'label': self._generate_cluster_label(results),
                    'results': results,
                    'size': len(results),
                    'coherence_score': float(self._calculate_cluster_coherence(results)),
                    'diversity_score': float(self._calculate_cluster_diversity(results))
                }
                clusters.append(cluster)
        
        return sorted(clusters, key=lambda x: x['size'], reverse=True)

    def _generate_cluster_label(self, results: List[Dict]) -> str:
        """Generate meaningful labels for clusters"""
        # Extract categories and titles
        categories = [result.get('category', 'general') for result in results]
        titles = [result['title'] for result in results]
        
        # Find most common category
        category_counts = defaultdict(int)
        for cat in categories:
            category_counts[cat] += 1
        
        most_common_category = max(category_counts.items(), key=lambda x: x[1])[0]
        
        # Generate label based on category
        category_labels = {
            'person_musician': 'Musicians & Artists',
            'person_politician': 'Political Figures',
            'person_artist': 'Artists & Creators',
            'location': 'Places & Locations',
            'company': 'Companies & Organizations',
            'product_tech': 'Technology Products',
            'programming': 'Programming & Development',
            'animal': 'Animals & Nature',
            'food': 'Food & Nutrition',
            'entertainment': 'Entertainment & Media',
            'education': 'Educational Content',
            'news': 'News & Current Events',
            'general': 'General Information'
        }
        
        return category_labels.get(most_common_category, 'Mixed Content')

    def _calculate_cluster_coherence(self, results: List[Dict]) -> float:
        """Calculate coherence score for a cluster"""
        try:
            if len(results) < 2:
                return 1.0
            
            embeddings = np.array([result['embedding'] for result in results])
            centroid = np.mean(embeddings, axis=0)
            
            # Calculate average distance to centroid
            distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
            avg_distance = np.mean(distances)
            
            # Convert to coherence score (lower distance = higher coherence)
            coherence = 1.0 / (1.0 + avg_distance)
            return min(coherence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating cluster coherence: {str(e)}")
            return 0.5

    def _calculate_cluster_diversity(self, results: List[Dict]) -> float:
        """Calculate diversity score for a cluster"""
        try:
            categories = [result.get('category', 'general') for result in results]
            unique_categories = len(set(categories))
            total_results = len(results)
            
            return unique_categories / total_results if total_results > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating cluster diversity: {str(e)}")
            return 0.5

    def process_feedback(self, feedback_data: Dict) -> Dict:
        """Process user feedback and update RL agent"""
        logger.info(f"Processing feedback: {feedback_data}")
        
        try:
            # Add timestamp and additional context
            feedback_data['timestamp'] = datetime.now().isoformat()
            feedback_data['query'] = self.query_history[-1]['query'] if self.query_history else ''
            
            # Store feedback
            self.user_feedback_history.append(feedback_data)
            self.system_metrics['total_feedback_items'] += 1
            
            # Update RL agent
            reward = self.rl_agent.process_feedback(feedback_data)
            self.system_metrics['total_reward'] += reward
            self.system_metrics['rl_episodes'] += 1
            
            # Update user satisfaction
            self._update_user_satisfaction()
            
            # Save system state periodically
            if self.system_metrics['total_feedback_items'] % 10 == 0:
                self.save_system_state()
            
            # Clean result for JSON serialization
            result = {
                'status': 'success',
                'reward': float(reward),
                'total_episodes': int(self.system_metrics['rl_episodes']),
                'exploration_rate': float(self.rl_agent.get_exploration_rate())
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'reward': 0.0,
                'total_episodes': self.system_metrics.get('rl_episodes', 0),
                'exploration_rate': 0.8
            }

    def _update_user_satisfaction(self):
        """Update average user satisfaction based on recent feedback"""
        try:
            if not self.user_feedback_history:
                return
            
            recent_feedback = self.user_feedback_history[-20:]  # Last 20 items
            positive_feedback = [
                'relevant', 'excellent', 'good', 'helpful', 'accurate'
            ]
            
            positive_count = sum(1 for fb in recent_feedback 
                               if fb.get('feedback') in positive_feedback)
            
            satisfaction = positive_count / len(recent_feedback)
            self.system_metrics['avg_user_satisfaction'] = satisfaction
            
        except Exception as e:
            logger.error(f"Error updating user satisfaction: {str(e)}")

    def get_system_metrics(self) -> Dict:
        """Get current system performance metrics"""
        try:
            if self.current_clusters:
                # Calculate clustering metrics
                cluster_labels = []
                true_labels = []  # Based on categories for evaluation
                
                for cluster in self.current_clusters:
                    for result in cluster['results']:
                        cluster_labels.append(cluster['id'])
                        true_labels.append(result.get('category', 'general'))
                
                metrics = self.metrics_calculator.calculate_all_metrics(
                    self.current_results,
                    self.current_clusters,
                    cluster_labels,
                    true_labels
                )
            else:
                metrics = {
                    'cluster_purity': 0.0,
                    'adjusted_rand_index': 0.0,
                    'silhouette_score': 0.0,
                    'normalized_mutual_info': 0.0
                }
            
            # Combine with system metrics
            metrics.update(self.system_metrics)
            metrics['user_satisfaction_pct'] = int(self.system_metrics['avg_user_satisfaction'] * 100)
            
            # Clean metrics for JSON serialization
            return clean_metrics_data(metrics)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            # Return basic metrics on error
            return clean_metrics_data({
                'total_queries': self.system_metrics.get('total_queries', 0),
                'total_feedback_items': self.system_metrics.get('total_feedback_items', 0),
                'user_satisfaction_pct': int(self.system_metrics.get('avg_user_satisfaction', 0.75) * 100),
                'rl_episodes': self.system_metrics.get('rl_episodes', 0),
                'total_reward': self.system_metrics.get('total_reward', 0.0),
                'cluster_purity': 0.0,
                'adjusted_rand_index': 0.0,
                'silhouette_score': 0.0,
                'normalized_mutual_info': 0.0
            })

# Initialize the system
system = SearchDisambiguationSystem()

# API Routes
@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for performing search"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        language = data.get('language', 'en')
        num_results = data.get('num_results', 20)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        results = system.perform_search(query, language, num_results)
        
        response_data = {
            'status': 'success',
            'results': results,
            'query': query,
            'language': language,
            'total_results': len(results)
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cluster', methods=['POST'])
def api_cluster():
    """API endpoint for clustering results"""
    try:
        data = request.get_json()
        algorithm = data.get('algorithm', 'kmeans')
        num_clusters = data.get('num_clusters', 4)
        min_cluster_size = data.get('min_cluster_size', 2)
        
        clusters = system.perform_clustering(algorithm, num_clusters, min_cluster_size)
        
        response_data = {
            'status': 'success',
            'clusters': clusters,
            'algorithm': algorithm,
            'total_clusters': len(clusters)
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in clustering: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """API endpoint for processing user feedback"""
    try:
        feedback_data = request.get_json()
        
        if not feedback_data:
            return jsonify({'error': 'Feedback data is required'}), 400
        
        result = system.process_feedback(feedback_data)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """API endpoint for getting system metrics"""
    try:
        metrics = system.get_system_metrics()
        return jsonify(metrics)
    
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    try:
        response_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'system_metrics': system.get_system_metrics()
        }
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return app.send_static_file('index.html')

if __name__ == '__main__':
    # Start background threads for system maintenance
    def periodic_save():
        while True:
            time.sleep(300)  # Save every 5 minutes
            try:
                system.save_system_state()
            except Exception as e:
                logger.error(f"Error in periodic save: {str(e)}")
    
    save_thread = threading.Thread(target=periodic_save, daemon=True)
    save_thread.start()
    
    logger.info("Starting Flask application...")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)