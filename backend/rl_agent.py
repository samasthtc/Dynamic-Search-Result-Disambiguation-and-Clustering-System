"""
Reinforcement Learning Agent for Dynamic Clustering Parameter Optimization
Implements Q-Learning as specified in the research paper
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import pickle
import os

logger = logging.getLogger(__name__)


class RLAgent:
    """
    Q-Learning agent for optimizing clustering parameters

    State: ⟨len, dens, spars, JS⟩ - query length, embedding density, TF-IDF sparsity, JS divergence
    Action: (representation, clustering) pair selection
    Reward: Silhouette + user feedback bonuses
    """

    def __init__(
        self,
        learning_rate: float = 0.2,
        discount_factor: float = 0.8,
        exploration_rate: float = 0.3,
        exploration_decay: float = 0.995,
    ):
        """
        Initialize RL agent with parameters from the paper

        Args:
            learning_rate: α = 0.2 (as specified in paper)
            discount_factor: γ = 0.8 (as specified in paper)
            exploration_rate: ε starting at 0.30 (as specified in paper)
            exploration_decay: ε decay to 0.05 over 300 episodes
        """

        # RL parameters from the research paper
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.05

        # Q-table for state-action values
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Episode tracking
        self.episodes = 0
        self.total_reward = 0.0
        self.episode_rewards = []

        # Action space as defined in paper
        self.algorithms = [
            "kmeans",
            "hdbscan",
            "bertopic",
            "gaussian_mixture",
            "hierarchical",
        ]
        self.representations = ["sentence_bert", "tfidf"]
        self.cluster_sizes = [2, 3, 4, 5, 6, 7, 8]

        # State discretization bins
        self.state_bins = {
            "query_length": [
                0,
                5,
                10,
                20,
                50,
            ],  # Very short, short, medium, long, very long
            "density": [0, 0.3, 0.6, 0.8, 1.0],  # Very sparse to very dense
            "sparsity": [0, 0.2, 0.5, 0.8, 1.0],  # Low to high sparsity
            "js_divergence": [0, 0.1, 0.3, 0.6, 1.0],  # Low to high divergence
        }

        # Load existing Q-table if available
        self.load_q_table()

        logger.info("RL Agent initialized with tabular Q-learning")

    def discretize_state(self, state: Dict[str, float]) -> Tuple[int, int, int, int]:
        """
        Discretize continuous state into bins for tabular Q-learning

        Args:
            state: Dictionary with continuous state values

        Returns:
            Tuple of discretized state indices
        """
        try:

            def find_bin(value: float, bins: List[float]) -> int:
                for i, threshold in enumerate(bins[1:], 1):
                    if value <= threshold:
                        return i - 1
                return len(bins) - 2

            query_len_bin = find_bin(
                state.get("query_length", 0), self.state_bins["query_length"]
            )
            density_bin = find_bin(state.get("density", 0), self.state_bins["density"])
            sparsity_bin = find_bin(
                state.get("sparsity", 0), self.state_bins["sparsity"]
            )
            js_bin = find_bin(
                state.get("js_divergence", 0), self.state_bins["js_divergence"]
            )

            return (query_len_bin, density_bin, sparsity_bin, js_bin)

        except Exception as e:
            logger.warning(f"Error discretizing state: {e}, using default")
            return (2, 2, 2, 2)  # Default middle bins

    def get_current_state(
        self, embeddings: np.ndarray, results: List[Dict]
    ) -> Dict[str, float]:
        """
        Extract state features as specified in the paper:
        State s = ⟨len, dens, spars, JS⟩

        Args:
            embeddings: Document embeddings
            results: Search results

        Returns:
            State dictionary
        """
        try:
            # Query length (from first result or average)
            if results:
                avg_query_len = np.mean(
                    [len(r.get("title", "").split()) for r in results]
                )
            else:
                avg_query_len = 5

            # Embedding density (mean pairwise cosine similarity)
            if len(embeddings) > 1:
                from sklearn.metrics.pairwise import cosine_similarity

                sim_matrix = cosine_similarity(embeddings)
                # Get upper triangle (excluding diagonal)
                upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
                density = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.5
            else:
                density = 0.5

            # TF-IDF sparsity (proportion of zero elements)
            sparsity = (
                0.3  # Default value (would need TF-IDF matrix for exact calculation)
            )

            # Jensen-Shannon divergence (approximated using embedding variance)
            if len(embeddings) > 1:
                js_divergence = np.mean(np.var(embeddings, axis=0))
                js_divergence = min(1.0, js_divergence)  # Normalize to [0,1]
            else:
                js_divergence = 0.5

            state = {
                "query_length": avg_query_len,
                "density": density,
                "sparsity": sparsity,
                "js_divergence": js_divergence,
            }

            return state

        except Exception as e:
            logger.warning(f"Error computing state: {e}, using defaults")
            return {
                "query_length": 5,
                "density": 0.5,
                "sparsity": 0.3,
                "js_divergence": 0.5,
            }

    def select_action(self, state: Dict[str, float]) -> Dict[str, Any]:
        """
        Select action using ε-greedy policy

        Args:
            state: Current state

        Returns:
            Action dictionary with algorithm and parameters
        """
        try:
            discrete_state = self.discretize_state(state)

            # ε-greedy action selection
            if np.random.random() < self.exploration_rate:
                # Explore: random action
                action = {
                    "algorithm": np.random.choice(self.algorithms),
                    "representation": np.random.choice(self.representations),
                    "num_clusters": np.random.choice(self.cluster_sizes),
                }
            else:
                # Exploit: best known action
                best_action = self._get_best_action(discrete_state)
                action = best_action

            return action

        except Exception as e:
            logger.warning(f"Error selecting action: {e}, using default")
            return {
                "algorithm": "bertopic",
                "representation": "sentence_bert",
                "num_clusters": 4,
            }

    def _get_best_action(
        self, discrete_state: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """Get the best action for a given state"""
        try:
            best_q_value = float("-inf")
            best_action = None

            # Check all possible actions
            for algorithm in self.algorithms:
                for representation in self.representations:
                    for num_clusters in self.cluster_sizes:
                        action_key = f"{algorithm}_{representation}_{num_clusters}"
                        q_value = self.q_table[discrete_state][action_key]

                        if q_value > best_q_value:
                            best_q_value = q_value
                            best_action = {
                                "algorithm": algorithm,
                                "representation": representation,
                                "num_clusters": num_clusters,
                            }

            # Default action if no best action found
            if best_action is None:
                best_action = {
                    "algorithm": "bertopic",
                    "representation": "sentence_bert",
                    "num_clusters": 4,
                }

            return best_action

        except Exception as e:
            logger.warning(f"Error getting best action: {e}")
            return {
                "algorithm": "bertopic",
                "representation": "sentence_bert",
                "num_clusters": 4,
            }

    def calculate_reward(self, feedback: str, context: str = "result") -> float:
        """
        Calculate reward based on user feedback as specified in the paper:
        Reward r = Silhouette + 0.1 if user accepts + 0.2 for merge/split

        Args:
            feedback: User feedback string
            context: Context of feedback ('result' or 'cluster')

        Returns:
            Reward value
        """
        try:
            base_reward = 0.0

            # Feedback-based rewards
            feedback_rewards = {
                # Result-level feedback
                "relevant": 0.1,
                "irrelevant": -0.1,
                "wrong_cluster": -0.05,
                # Cluster-level feedback
                "excellent": 0.2,
                "good": 0.1,
                "poor": -0.1,
                "should_split": 0.2,  # Active improvement feedback
                "should_merge": 0.2,  # Active improvement feedback
            }

            reward = base_reward + feedback_rewards.get(feedback, 0.0)

            # Context-specific adjustments
            if context == "cluster" and feedback in ["excellent", "good"]:
                reward *= 1.5  # Boost cluster-level positive feedback

            return reward

        except Exception as e:
            logger.warning(f"Error calculating reward: {e}")
            return 0.0

    def update_policy(
        self,
        state: Dict[str, float],
        reward: float,
        feedback: str,
        action: Dict[str, Any] = None,
    ):
        """
        Update Q-table using Q-learning update rule

        Args:
            state: Current state
            reward: Reward received
            feedback: User feedback
            action: Action taken (if None, uses last action)
        """
        try:
            discrete_state = self.discretize_state(state)

            # Use default action if none provided
            if action is None:
                action = {
                    "algorithm": "bertopic",
                    "representation": "sentence_bert",
                    "num_clusters": 4,
                }

            action_key = f"{action['algorithm']}_{action['representation']}_{action['num_clusters']}"

            # Current Q-value
            current_q = self.q_table[discrete_state][action_key]

            # Get maximum Q-value for next state (assume same state for simplicity)
            max_next_q = (
                max(self.q_table[discrete_state].values())
                if self.q_table[discrete_state]
                else 0.0
            )

            # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ * max_a' Q(s',a') - Q(s,a)]
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )

            # Update Q-table
            self.q_table[discrete_state][action_key] = new_q

            # Update episode tracking
            self.episodes += 1
            self.total_reward += reward
            self.episode_rewards.append(reward)

            # Decay exploration rate
            if self.exploration_rate > self.min_exploration_rate:
                self.exploration_rate *= self.exploration_decay

            # Save Q-table periodically
            if self.episodes % 10 == 0:
                self.save_q_table()

            logger.debug(
                f"Updated Q-value: {current_q:.3f} -> {new_q:.3f}, reward: {reward:.3f}"
            )

        except Exception as e:
            logger.error(f"Error updating policy: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current RL agent status"""
        return {
            "episodes": self.episodes,
            "total_reward": self.total_reward,
            "exploration_rate": self.exploration_rate,
            "q_table_size": len(self.q_table),
            "avg_reward": (
                np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            ),
            "recent_performance": (
                np.mean(self.episode_rewards[-10:])
                if len(self.episode_rewards) >= 10
                else 0.0
            ),
        }

    def save_q_table(self, filepath: str = "q_table.pkl"):
        """Save Q-table to disk"""
        try:
            # Convert defaultdict to regular dict for saving
            q_table_dict = {
                state: dict(actions) for state, actions in self.q_table.items()
            }

            data = {
                "q_table": q_table_dict,
                "episodes": self.episodes,
                "total_reward": self.total_reward,
                "exploration_rate": self.exploration_rate,
                "episode_rewards": self.episode_rewards,
            }

            with open(filepath, "wb") as f:
                pickle.dump(data, f)

            logger.debug(f"Q-table saved to {filepath}")

        except Exception as e:
            logger.warning(f"Error saving Q-table: {e}")

    def load_q_table(self, filepath: str = "q_table.pkl"):
        """Load Q-table from disk"""
        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    data = pickle.load(f)

                # Convert back to defaultdict
                self.q_table = defaultdict(lambda: defaultdict(float))
                for state, actions in data.get("q_table", {}).items():
                    for action, q_value in actions.items():
                        self.q_table[state][action] = q_value

                self.episodes = data.get("episodes", 0)
                self.total_reward = data.get("total_reward", 0.0)
                self.exploration_rate = data.get(
                    "exploration_rate", self.exploration_rate
                )
                self.episode_rewards = data.get("episode_rewards", [])

                logger.info(
                    f"Q-table loaded from {filepath}, episodes: {self.episodes}"
                )

        except Exception as e:
            logger.info(f"No existing Q-table found or error loading: {e}")

    def reset_learning(self):
        """Reset the learning process"""
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.episodes = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.exploration_rate = 0.3  # Reset to initial value

        logger.info("RL agent reset")

    def get_policy_summary(self) -> Dict[str, Any]:
        """Get a summary of the learned policy"""
        try:
            if not self.q_table:
                return {"message": "No policy learned yet"}

            # Find best action for each state
            policy = {}
            for state, actions in self.q_table.items():
                if actions:
                    best_action = max(actions.items(), key=lambda x: x[1])
                    policy[str(state)] = {
                        "best_action": best_action[0],
                        "q_value": best_action[1],
                        "actions_explored": len(actions),
                    }

            # Algorithm preferences
            algorithm_counts = defaultdict(int)
            for state_policy in policy.values():
                algorithm = state_policy["best_action"].split("_")[0]
                algorithm_counts[algorithm] += 1

            return {
                "states_explored": len(policy),
                "algorithm_preferences": dict(algorithm_counts),
                "total_state_action_pairs": sum(
                    len(actions) for actions in self.q_table.values()
                ),
                "average_q_value": (
                    np.mean(
                        [
                            max(actions.values())
                            for actions in self.q_table.values()
                            if actions
                        ]
                    )
                    if self.q_table
                    else 0.0
                ),
            }

        except Exception as e:
            logger.error(f"Error getting policy summary: {e}")
            return {"error": str(e)}
