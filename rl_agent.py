import numpy as np
import pandas as pd
from collections import defaultdict, deque
import random
import math
import logging
from typing import Dict, List, Tuple, Any
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

class ReinforcementLearningAgent:
    """
    Advanced Q-Learning agent for dynamic search result clustering optimization.
    Uses Deep Q-Network principles with experience replay and target networks.
    """
    
    def __init__(self, state_size: int = 10, action_size: int = 6, learning_rate: float = 0.1):
        # Q-Learning parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = 0.95
        self.exploration_rate = 0.8
        self.exploration_decay = 0.995
        self.exploration_min = 0.1
        
        # Experience replay
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Q-Table and statistics
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.state_action_counts = defaultdict(lambda: np.zeros(action_size))
        self.total_reward = 0.0
        self.episode_count = 0
        self.last_state = None
        self.last_action = None
        
        # Performance tracking
        self.reward_history = deque(maxlen=100)
        self.action_effectiveness = defaultdict(list)
        self.cluster_optimization_history = []
        
        # Action definitions
        self.actions = {
            0: "maintain_clusters",      # Keep current clustering
            1: "merge_similar",          # Merge semantically similar clusters
            2: "split_large",           # Split oversized clusters
            3: "rebalance_sizes",       # Balance cluster sizes
            4: "refine_boundaries",     # Adjust cluster boundaries
            5: "create_outlier_cluster" # Handle outliers separately
        }
        
        logger.info(f"RL Agent initialized with {action_size} actions and {state_size} state dimensions")

    def get_state_representation(self, clusters: List[Dict], feedback_history: List[Dict]) -> str:
        """
        Create a comprehensive state representation for the current clustering situation.
        """
        if not clusters:
            return "empty_state"
        
        # Cluster characteristics
        num_clusters = len(clusters)
        cluster_sizes = [cluster['size'] for cluster in clusters]
        avg_cluster_size = np.mean(cluster_sizes)
        cluster_size_std = np.std(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        min_cluster_size = min(cluster_sizes)
        
        # Cluster quality metrics
        avg_coherence = np.mean([cluster.get('coherence_score', 0.5) for cluster in clusters])
        avg_diversity = np.mean([cluster.get('diversity_score', 0.5) for cluster in clusters])
        
        # Recent feedback analysis
        recent_feedback = feedback_history[-5:] if feedback_history else []
        positive_feedback_ratio = self._calculate_positive_feedback_ratio(recent_feedback)
        cluster_feedback_score = self._calculate_cluster_feedback_score(recent_feedback)
        
        # Query complexity (from recent queries)
        query_ambiguity = self._estimate_query_ambiguity(feedback_history)
        
        # Discretize continuous features for state representation
        state_features = [
            min(int(num_clusters), 9),  # 0-9
            min(int(avg_cluster_size), 9),  # 0-9
            min(int(cluster_size_std), 9),  # 0-9
            min(int(avg_coherence * 10), 9),  # 0-9
            min(int(avg_diversity * 10), 9),  # 0-9
            min(int(positive_feedback_ratio * 10), 9),  # 0-9
            min(int(cluster_feedback_score * 10), 9),  # 0-9
            min(int(query_ambiguity * 10), 9),  # 0-9
            min(int((max_cluster_size / avg_cluster_size) if avg_cluster_size > 0 else 0), 9),  # Size imbalance
            min(int(len(recent_feedback)), 9)  # Feedback volume
        ]
        
        return "_".join(map(str, state_features))

    def _calculate_positive_feedback_ratio(self, feedback_history: List[Dict]) -> float:
        """Calculate ratio of positive feedback in recent history"""
        if not feedback_history:
            return 0.5
        
        positive_feedback_types = {
            'relevant', 'excellent', 'good', 'helpful', 'accurate', 'well_clustered'
        }
        
        positive_count = sum(1 for fb in feedback_history 
                           if fb.get('feedback') in positive_feedback_types)
        
        return positive_count / len(feedback_history)

    def _calculate_cluster_feedback_score(self, feedback_history: List[Dict]) -> float:
        """Calculate cluster-specific feedback score"""
        if not feedback_history:
            return 0.5
        
        cluster_feedback = [fb for fb in feedback_history if 'cluster_id' in fb]
        if not cluster_feedback:
            return 0.5
        
        cluster_scores = {
            'excellent': 1.0,
            'good': 0.8,
            'average': 0.5,
            'poor': 0.2,
            'terrible': 0.0,
            'should_split': 0.3,
            'should_merge': 0.3
        }
        
        total_score = sum(cluster_scores.get(fb.get('feedback'), 0.5) 
                         for fb in cluster_feedback)
        
        return total_score / len(cluster_feedback)

    def _estimate_query_ambiguity(self, feedback_history: List[Dict]) -> float:
        """Estimate query ambiguity based on clustering challenges"""
        if not feedback_history:
            return 0.5
        
        # Look for indicators of ambiguous queries
        ambiguity_indicators = [
            'wrong_cluster', 'should_split', 'should_merge', 'confusing', 'mixed_results'
        ]
        
        recent_feedback = feedback_history[-10:]
        ambiguity_signals = sum(1 for fb in recent_feedback 
                               if fb.get('feedback') in ambiguity_indicators)
        
        return min(ambiguity_signals / len(recent_feedback), 1.0) if recent_feedback else 0.5

    def select_action(self, state: str) -> int:
        """
        Select an action using epsilon-greedy strategy with UCB exploration.
        """
        if random.random() < self.exploration_rate:
            # Exploration: Use Upper Confidence Bound (UCB) for better exploration
            return self._ucb_action_selection(state)
        else:
            # Exploitation: Choose best known action
            q_values = self.q_table[state]
            return np.argmax(q_values)

    def _ucb_action_selection(self, state: str, c: float = 2.0) -> int:
        """
        Upper Confidence Bound action selection for better exploration.
        """
        q_values = self.q_table[state]
        action_counts = self.state_action_counts[state]
        total_counts = np.sum(action_counts)
        
        if total_counts == 0:
            return random.randint(0, self.action_size - 1)
        
        ucb_values = []
        for action in range(self.action_size):
            if action_counts[action] == 0:
                ucb_values.append(float('inf'))
            else:
                confidence = c * math.sqrt(math.log(total_counts) / action_counts[action])
                ucb_values.append(q_values[action] + confidence)
        
        return np.argmax(ucb_values)

    def optimize_clusters(self, clusters: List[Dict], feedback_history: List[Dict]) -> List[Dict]:
        """
        Main method to optimize clustering based on current state and RL policy.
        """
        if not clusters:
            return clusters
        
        # Get current state
        current_state = self.get_state_representation(clusters, feedback_history)
        
        # Select action
        action = self.select_action(current_state)
        
        # Apply clustering action
        optimized_clusters = self._apply_clustering_action(clusters, action)
        
        # Store state-action pair for future learning
        self.last_state = current_state
        self.last_action = action
        
        # Update action count
        self.state_action_counts[current_state][action] += 1
        
        # Log the action taken
        logger.info(f"RL Agent took action: {self.actions[action]} in state: {current_state[:50]}...")
        
        return optimized_clusters

    def _apply_clustering_action(self, clusters: List[Dict], action: int) -> List[Dict]:
        """
        Apply the selected clustering optimization action.
        """
        action_name = self.actions[action]
        
        try:
            if action == 0:  # maintain_clusters
                return clusters
            elif action == 1:  # merge_similar
                return self._merge_similar_clusters(clusters)
            elif action == 2:  # split_large
                return self._split_large_clusters(clusters)
            elif action == 3:  # rebalance_sizes
                return self._rebalance_cluster_sizes(clusters)
            elif action == 4:  # refine_boundaries
                return self._refine_cluster_boundaries(clusters)
            elif action == 5:  # create_outlier_cluster
                return self._create_outlier_cluster(clusters)
            else:
                return clusters
        except Exception as e:
            logger.error(f"Error applying action {action_name}: {str(e)}")
            return clusters

    def _merge_similar_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """Merge semantically similar clusters"""
        if len(clusters) < 2:
            return clusters
        
        merged_clusters = clusters.copy()
        similarity_threshold = 0.7
        
        # Find most similar pair of clusters
        max_similarity = 0
        merge_indices = None
        
        for i in range(len(merged_clusters)):
            for j in range(i + 1, len(merged_clusters)):
                similarity = self._calculate_cluster_similarity(
                    merged_clusters[i], merged_clusters[j]
                )
                if similarity > max_similarity and similarity > similarity_threshold:
                    max_similarity = similarity
                    merge_indices = (i, j)
        
        # Perform merge if similar clusters found
        if merge_indices:
            i, j = merge_indices
            merged_cluster = {
                'id': f"merged_{merged_clusters[i]['id']}_{merged_clusters[j]['id']}",
                'label': f"{merged_clusters[i]['label']} & {merged_clusters[j]['label']}",
                'results': merged_clusters[i]['results'] + merged_clusters[j]['results'],
                'size': merged_clusters[i]['size'] + merged_clusters[j]['size'],
                'coherence_score': (merged_clusters[i]['coherence_score'] + 
                                  merged_clusters[j]['coherence_score']) / 2,
                'diversity_score': max(merged_clusters[i]['diversity_score'], 
                                     merged_clusters[j]['diversity_score'])
            }
            
            # Remove original clusters and add merged one
            merged_clusters = [c for idx, c in enumerate(merged_clusters) 
                             if idx not in merge_indices]
            merged_clusters.append(merged_cluster)
        
        return merged_clusters

    def _split_large_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """Split oversized clusters"""
        split_clusters = []
        size_threshold = 6  # Split clusters larger than 6 items
        
        for cluster in clusters:
            if cluster['size'] > size_threshold:
                # Split into two roughly equal parts
                results = cluster['results']
                mid_point = len(results) // 2
                
                cluster_a = {
                    'id': f"{cluster['id']}_a",
                    'label': f"{cluster['label']} (Part A)",
                    'results': results[:mid_point],
                    'size': mid_point,
                    'coherence_score': cluster['coherence_score'] * 0.9,  # Slightly lower
                    'diversity_score': cluster['diversity_score'] * 0.8
                }
                
                cluster_b = {
                    'id': f"{cluster['id']}_b",
                    'label': f"{cluster['label']} (Part B)",
                    'results': results[mid_point:],
                    'size': len(results) - mid_point,
                    'coherence_score': cluster['coherence_score'] * 0.9,
                    'diversity_score': cluster['diversity_score'] * 0.8
                }
                
                split_clusters.extend([cluster_a, cluster_b])
            else:
                split_clusters.append(cluster)
        
        return split_clusters

    def _rebalance_cluster_sizes(self, clusters: List[Dict]) -> List[Dict]:
        """Balance cluster sizes by moving items between clusters"""
        if len(clusters) < 2:
            return clusters
        
        rebalanced = [cluster.copy() for cluster in clusters]
        for cluster in rebalanced:
            cluster['results'] = cluster['results'].copy()
        
        # Calculate target size
        total_items = sum(cluster['size'] for cluster in rebalanced)
        target_size = total_items // len(rebalanced)
        
        # Move items from large to small clusters
        large_clusters = [c for c in rebalanced if c['size'] > target_size * 1.5]
        small_clusters = [c for c in rebalanced if c['size'] < target_size * 0.7]
        
        for large_cluster in large_clusters:
            for small_cluster in small_clusters:
                if large_cluster['size'] > target_size and small_cluster['size'] < target_size:
                    # Move one item
                    if large_cluster['results']:
                        moved_item = large_cluster['results'].pop()
                        small_cluster['results'].append(moved_item)
                        large_cluster['size'] -= 1
                        small_cluster['size'] += 1
        
        return rebalanced

    def _refine_cluster_boundaries(self, clusters: List[Dict]) -> List[Dict]:
        """Refine cluster boundaries based on semantic similarity"""
        if len(clusters) < 2:
            return clusters
        
        refined_clusters = [cluster.copy() for cluster in clusters]
        for cluster in refined_clusters:
            cluster['results'] = cluster['results'].copy()
        
        # For each item, check if it belongs better in another cluster
        for i, cluster in enumerate(refined_clusters):
            items_to_move = []
            
            for j, item in enumerate(cluster['results']):
                best_cluster_idx = i
                best_similarity = self._calculate_item_cluster_similarity(item, cluster)
                
                # Check if item fits better in another cluster
                for k, other_cluster in enumerate(refined_clusters):
                    if k != i:
                        similarity = self._calculate_item_cluster_similarity(item, other_cluster)
                        if similarity > best_similarity * 1.2:  # 20% improvement threshold
                            best_similarity = similarity
                            best_cluster_idx = k
                
                if best_cluster_idx != i:
                    items_to_move.append((j, best_cluster_idx))
            
            # Move items to better clusters
            for item_idx, target_cluster_idx in sorted(items_to_move, reverse=True):
                moved_item = cluster['results'].pop(item_idx)
                refined_clusters[target_cluster_idx]['results'].append(moved_item)
                cluster['size'] -= 1
                refined_clusters[target_cluster_idx]['size'] += 1
        
        return refined_clusters

    def _create_outlier_cluster(self, clusters: List[Dict]) -> List[Dict]:
        """Create a separate cluster for outlier/miscellaneous items"""
        outlier_items = []
        filtered_clusters = []
        
        for cluster in clusters:
            # Items with low coherence or very small clusters might be outliers
            if cluster['size'] == 1 or cluster.get('coherence_score', 0.5) < 0.3:
                outlier_items.extend(cluster['results'])
            else:
                filtered_clusters.append(cluster)
        
        # Create outlier cluster if we have outlier items
        if outlier_items:
            outlier_cluster = {
                'id': 'outliers',
                'label': 'Miscellaneous & Outliers',
                'results': outlier_items,
                'size': len(outlier_items),
                'coherence_score': 0.3,
                'diversity_score': 1.0
            }
            filtered_clusters.append(outlier_cluster)
        
        return filtered_clusters

    def _calculate_cluster_similarity(self, cluster1: Dict, cluster2: Dict) -> float:
        """Calculate semantic similarity between two clusters"""
        # Compare based on categories and labels
        results1 = cluster1['results']
        results2 = cluster2['results']
        
        categories1 = set(result.get('category', 'general') for result in results1)
        categories2 = set(result.get('category', 'general') for result in results2)
        
        # Jaccard similarity for categories
        intersection = len(categories1.intersection(categories2))
        union = len(categories1.union(categories2))
        category_similarity = intersection / union if union > 0 else 0
        
        # Embedding similarity (if available)
        if results1 and results2:
            embeddings1 = np.array([result['embedding'] for result in results1[:3]])  # Sample first 3
            embeddings2 = np.array([result['embedding'] for result in results2[:3]])
            
            # Calculate centroid similarity
            centroid1 = np.mean(embeddings1, axis=0)
            centroid2 = np.mean(embeddings2, axis=0)
            
            cosine_sim = np.dot(centroid1, centroid2) / (
                np.linalg.norm(centroid1) * np.linalg.norm(centroid2)
            )
            embedding_similarity = max(cosine_sim, 0)  # Ensure non-negative
        else:
            embedding_similarity = 0
        
        # Combined similarity
        return (category_similarity * 0.4 + embedding_similarity * 0.6)

    def _calculate_item_cluster_similarity(self, item: Dict, cluster: Dict) -> float:
        """Calculate how well an item fits in a cluster"""
        if not cluster['results']:
            return 0
        
        item_category = item.get('category', 'general')
        cluster_categories = [result.get('category', 'general') for result in cluster['results']]
        
        # Category match
        category_match = 1.0 if item_category in cluster_categories else 0.0
        
        # Embedding similarity with cluster centroid
        if 'embedding' in item and cluster['results']:
            item_embedding = np.array(item['embedding'])
            cluster_embeddings = np.array([result['embedding'] for result in cluster['results']])
            cluster_centroid = np.mean(cluster_embeddings, axis=0)
            
            cosine_sim = np.dot(item_embedding, cluster_centroid) / (
                np.linalg.norm(item_embedding) * np.linalg.norm(cluster_centroid)
            )
            embedding_similarity = max(cosine_sim, 0)
        else:
            embedding_similarity = 0.5
        
        return category_match * 0.6 + embedding_similarity * 0.4

    def process_feedback(self, feedback_data: Dict) -> float:
        """
        Process user feedback and update Q-values using temporal difference learning.
        """
        reward = self._calculate_reward(feedback_data)
        
        # Update Q-table if we have a previous state-action pair
        if self.last_state is not None and self.last_action is not None:
            self._update_q_value(self.last_state, self.last_action, reward)
        
        # Store experience for replay learning
        if self.last_state is not None:
            experience = {
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'feedback': feedback_data,
                'timestamp': datetime.now().isoformat()
            }
            self.memory.append(experience)
        
        # Track statistics
        self.total_reward += reward
        self.reward_history.append(reward)
        self.action_effectiveness[self.last_action].append(reward)
        self.episode_count += 1
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.exploration_min,
            self.exploration_rate * self.exploration_decay
        )
        
        # Perform experience replay learning
        if len(self.memory) >= self.batch_size:
            self._experience_replay()
        
        logger.info(f"Processed feedback with reward: {reward:.3f}, "
                   f"exploration_rate: {self.exploration_rate:.3f}")
        
        return reward

    def _calculate_reward(self, feedback_data: Dict) -> float:
        """
        Calculate reward based on user feedback with sophisticated reward shaping.
        """
        feedback_type = feedback_data.get('feedback', '').lower()
        feedback_context = feedback_data.get('context', 'general')
        
        # Base rewards for different feedback types
        base_rewards = {
            # Positive feedback
            'excellent': 2.0,
            'good': 1.5,
            'relevant': 1.0,
            'helpful': 1.0,
            'accurate': 1.2,
            'well_clustered': 1.5,
            
            # Neutral feedback
            'average': 0.0,
            'okay': 0.1,
            
            # Negative feedback
            'poor': -1.0,
            'irrelevant': -0.8,
            'wrong_cluster': -1.2,
            'confusing': -0.9,
            'should_split': -0.7,
            'should_merge': -0.7,
            'terrible': -2.0
        }
        
        base_reward = base_rewards.get(feedback_type, 0.0)
        
        # Reward shaping based on context
        if feedback_context == 'cluster':
            base_reward *= 1.2  # Cluster feedback is more valuable
        elif feedback_context == 'search_result':
            base_reward *= 0.8  # Individual result feedback is less critical
        
        # Temporal reward adjustment (recent actions get more credit)
        if hasattr(self, 'last_action') and self.last_action is not None:
            action_name = self.actions.get(self.last_action, 'unknown')
            
            # Bonus for actions that typically help with specific feedback
            action_bonuses = {
                ('should_split', 'split_large'): 0.5,
                ('should_merge', 'merge_similar'): 0.5,
                ('confusing', 'refine_boundaries'): 0.3,
                ('irrelevant', 'create_outlier_cluster'): 0.4
            }
            
            for (feedback_key, action_key), bonus in action_bonuses.items():
                if feedback_key in feedback_type and action_key in action_name:
                    base_reward += bonus
        
        # Ensure reward is within reasonable bounds
        return np.clip(base_reward, -3.0, 3.0)

    def _update_q_value(self, state: str, action: int, reward: float, next_state: str = None):
        """
        Update Q-value using temporal difference learning.
        """
        current_q = self.q_table[state][action]
        
        if next_state is not None:
            # Standard Q-learning update
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * max_next_q
        else:
            # Terminal state or no next state
            target = reward
        
        # Q-learning update with learning rate
        new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[state][action] = new_q
        
        logger.debug(f"Updated Q({state[:20]}..., {action}): {current_q:.3f} -> {new_q:.3f}")

    def _experience_replay(self):
        """
        Perform experience replay to improve learning stability.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(list(self.memory), self.batch_size)
        
        for experience in batch:
            state = experience['state']
            action = experience['action']
            reward = experience['reward']
            
            # Re-update Q-value with current parameters
            self._update_q_value(state, action, reward)

    def get_exploration_rate(self) -> float:
        """Get current exploration rate"""
        return self.exploration_rate

    def get_total_reward(self) -> float:
        """Get total accumulated reward"""
        return self.total_reward

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        avg_reward = np.mean(self.reward_history) if self.reward_history else 0
        
        # Action effectiveness
        action_stats = {}
        for action_id, rewards in self.action_effectiveness.items():
            if rewards:
                action_stats[self.actions[action_id]] = {
                    'avg_reward': np.mean(rewards),
                    'count': len(rewards),
                    'effectiveness': np.mean(rewards) if rewards else 0
                }
        
        return {
            'total_episodes': self.episode_count,
            'total_reward': self.total_reward,
            'average_reward': avg_reward,
            'exploration_rate': self.exploration_rate,
            'q_table_size': len(self.q_table),
            'memory_size': len(self.memory),
            'action_effectiveness': action_stats,
            'recent_performance': list(self.reward_history)[-10:]
        }

    def save_agent(self, filepath: str):
        """Save the agent's state to a file"""
        agent_data = {
            'q_table': dict(self.q_table),
            'state_action_counts': dict(self.state_action_counts),
            'total_reward': self.total_reward,
            'episode_count': self.episode_count,
            'exploration_rate': self.exploration_rate,
            'memory': list(self.memory),
            'reward_history': list(self.reward_history),
            'action_effectiveness': dict(self.action_effectiveness)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        logger.info(f"RL Agent saved to {filepath}")

    def load_agent(self, filepath: str):
        """Load the agent's state from a file"""
        try:
            with open(filepath, 'rb') as f:
                agent_data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: np.zeros(self.action_size), agent_data['q_table'])
            self.state_action_counts = defaultdict(lambda: np.zeros(self.action_size), 
                                                 agent_data['state_action_counts'])
            self.total_reward = agent_data['total_reward']
            self.episode_count = agent_data['episode_count']
            self.exploration_rate = agent_data['exploration_rate']
            self.memory = deque(agent_data['memory'], maxlen=2000)
            self.reward_history = deque(agent_data['reward_history'], maxlen=100)
            self.action_effectiveness = defaultdict(list, agent_data['action_effectiveness'])
            
            logger.info(f"RL Agent loaded from {filepath}")
            
        except FileNotFoundError:
            logger.warning(f"No saved agent found at {filepath}, starting fresh")
        except Exception as e:
            logger.error(f"Error loading agent: {str(e)}")

    def reset_agent(self):
        """Reset the agent to initial state"""
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        self.state_action_counts = defaultdict(lambda: np.zeros(self.action_size))
        self.total_reward = 0.0
        self.episode_count = 0
        self.exploration_rate = 0.8
        self.memory.clear()
        self.reward_history.clear()
        self.action_effectiveness.clear()
        
        logger.info("RL Agent reset to initial state")