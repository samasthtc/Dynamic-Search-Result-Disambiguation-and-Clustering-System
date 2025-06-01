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

# Custom imports for our components
from rl_agent import ReinforcementLearningAgent
from clustering_algorithms import ClusteringManager
from search_simulator import SearchSimulator
from evaluation_metrics import MetricsCalculator
from arabic_processor import ArabicTextProcessor

app = Flask(__name__)
CORS(app)

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
        """Load any pre-trained models or saved state"""
        try:
            with open('rl_agent_state.pkl', 'rb') as f:
                self.rl_agent = pickle.load(f)
            logger.info("Loaded pre-trained RL agent")
        except FileNotFoundError:
            logger.info("No pre-trained RL agent found, starting fresh")

    def save_system_state(self):
        """Save system state for persistence"""
        with open('rl_agent_state.pkl', 'wb') as f:
            pickle.dump(self.rl_agent, f)
        
        with open('system_metrics.json', 'w') as f:
            json.dump(self.system_metrics, f)

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
        
        # Use search simulator to generate realistic results
        results = self.search_simulator.simulate_search(query, language, num_results)
        
        # Generate embeddings for results
        texts = [f"{result['title']} {result['snippet']}" for result in results]
        embeddings = self.generate_embeddings(texts, language)
        
        # Add embeddings to results
        for i, result in enumerate(results):
            result['embedding'] = embeddings[i].tolist()
            result['relevance_score'] = np.random.uniform(0.3, 1.0)  # Simulated relevance
        
        self.current_results = results
        self.query_history.append({
            'query': query,
            'language': language,
            'timestamp': datetime.now().isoformat(),
            'num_results': len(results)
        })
        
        self.system_metrics['total_queries'] += 1
        return results

    def perform_clustering(self, algorithm: str = 'kmeans', num_clusters: int = 4, 
                          min_cluster_size: int = 2) -> List[Dict]:
        """Perform clustering on current search results"""
        if not self.current_results:
            return []
        
        logger.info(f"Performing clustering with algorithm: {algorithm}")
        
        # Extract embeddings
        embeddings = np.array([result['embedding'] for result in self.current_results])
        
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
        
        self.current_clusters = optimized_clusters
        return optimized_clusters

    def _organize_clusters(self, cluster_labels: np.ndarray) -> List[Dict]:
        """Organize search results into cluster structure"""
        cluster_dict = defaultdict(list)
        
        for i, label in enumerate(cluster_labels):
            if label != -1:  # -1 indicates noise/outlier in some algorithms
                cluster_dict[label].append(self.current_results[i])
        
        clusters = []
        for cluster_id, results in cluster_dict.items():
            if len(results) > 0:
                cluster = {
                    'id': cluster_id,
                    'label': self._generate_cluster_label(results),
                    'results': results,
                    'size': len(results),
                    'coherence_score': self._calculate_cluster_coherence(results),
                    'diversity_score': self._calculate_cluster_diversity(results)
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

    def _calculate_cluster_diversity(self, results: List[Dict]) -> float:
        """Calculate diversity score for a cluster"""
        categories = [result.get('category', 'general') for result in results]
        unique_categories = len(set(categories))
        total_results = len(results)
        
        return unique_categories / total_results if total_results > 0 else 0

    def process_feedback(self, feedback_data: Dict) -> Dict:
        """Process user feedback and update RL agent"""
        logger.info(f"Processing feedback: {feedback_data}")
        
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
        
        return {
            'status': 'success',
            'reward': reward,
            'total_episodes': self.system_metrics['rl_episodes'],
            'exploration_rate': self.rl_agent.get_exploration_rate()
        }

    def _update_user_satisfaction(self):
        """Update average user satisfaction based on recent feedback"""
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

    def get_system_metrics(self) -> Dict:
        """Get current system performance metrics"""
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
        
        return metrics

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
        
        return jsonify({
            'status': 'success',
            'results': results,
            'query': query,
            'language': language,
            'total_results': len(results)
        })
    
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
        
        return jsonify({
            'status': 'success',
            'clusters': clusters,
            'algorithm': algorithm,
            'total_clusters': len(clusters)
        })
    
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
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system_metrics': system.get_system_metrics()
    })

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return app.send_static_file('index.html')

if __name__ == '__main__':
    # Start background threads for system maintenance
    def periodic_save():
        while True:
            time.sleep(300)  # Save every 5 minutes
            system.save_system_state()
    
    save_thread = threading.Thread(target=periodic_save, daemon=True)
    save_thread.start()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)