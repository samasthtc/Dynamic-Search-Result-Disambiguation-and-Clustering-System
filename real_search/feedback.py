"""
Feedback Processor
Handles user feedback and reinforcement learning for cluster optimization
"""

import logging
import pickle
import json
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackProcessor:
    """
    Processes user feedback and applies reinforcement learning for system improvement
    """

    def __init__(self):
        # Feedback storage
        self.feedback_history = deque(maxlen=1000)
        self.cluster_feedback = defaultdict(list)

        # RL-like optimization
        self.action_rewards = defaultdict(list)
        self.cluster_optimization_stats = {
            "merge_attempts": 0,
            "split_attempts": 0,
            "rebalance_attempts": 0,
            "total_rewards": 0.0,
            "episodes": 0,
        }

        # User satisfaction tracking
        self.satisfaction_history = deque(maxlen=100)

        logger.info("Feedback Processor initialized")

    def process_feedback(self, feedback_data: Dict) -> Dict:
        """
        Process user feedback and update system knowledge

        Args:
            feedback_data: User feedback information

        Returns:
            Processing result with reward information
        """
        try:
            # Store feedback
            self.feedback_history.append(feedback_data)

            # Calculate reward
            reward = self._calculate_reward(feedback_data)

            # Update satisfaction tracking
            self._update_satisfaction(feedback_data)

            # Update optimization stats
            self.cluster_optimization_stats["total_rewards"] += reward
            self.cluster_optimization_stats["episodes"] += 1

            # Store cluster-specific feedback if applicable
            if "cluster_id" in feedback_data:
                self.cluster_feedback[feedback_data["cluster_id"]].append(
                    {
                        "feedback": feedback_data["feedback"],
                        "reward": reward,
                        "timestamp": feedback_data["timestamp"],
                    }
                )

            logger.debug(f"Processed feedback with reward: {reward}")

            return {
                "status": "success",
                "reward": reward,
                "total_episodes": self.cluster_optimization_stats["episodes"],
                "avg_satisfaction": self._get_avg_satisfaction(),
            }

        except Exception as e:
            logger.error(f"Feedback processing error: {str(e)}")
            return {"status": "error", "error": str(e), "reward": 0.0}

    def _calculate_reward(self, feedback_data: Dict) -> float:
        """Calculate reward based on feedback type and context"""
        feedback_type = feedback_data.get("feedback", "").lower()
        context = feedback_data.get("context", "general")

        # Base rewards for different feedback types
        reward_map = {
            "excellent": 2.0,
            "good": 1.5,
            "relevant": 1.0,
            "helpful": 1.0,
            "accurate": 1.2,
            "well_clustered": 1.5,
            "average": 0.0,
            "okay": 0.1,
            "poor": -1.0,
            "irrelevant": -0.8,
            "wrong_cluster": -1.2,
            "confusing": -0.9,
            "should_split": -0.7,
            "should_merge": -0.7,
            "terrible": -2.0,
        }

        base_reward = reward_map.get(feedback_type, 0.0)

        # Context modifiers
        if context == "cluster":
            base_reward *= 1.2  # Cluster feedback is more valuable
        elif context == "result":
            base_reward *= 0.8  # Individual result feedback is less critical

        # Temporal bonus for recent feedback
        reward = base_reward * 1.1  # Slight recency bonus

        return np.clip(reward, -3.0, 3.0)

    def _update_satisfaction(self, feedback_data: Dict):
        """Update user satisfaction tracking"""
        feedback_type = feedback_data.get("feedback", "").lower()

        # Map feedback to satisfaction score (0-1)
        satisfaction_map = {
            "excellent": 1.0,
            "good": 0.8,
            "relevant": 0.7,
            "helpful": 0.7,
            "accurate": 0.8,
            "well_clustered": 0.9,
            "average": 0.5,
            "okay": 0.4,
            "poor": 0.2,
            "irrelevant": 0.1,
            "wrong_cluster": 0.1,
            "confusing": 0.2,
            "terrible": 0.0,
        }

        satisfaction = satisfaction_map.get(feedback_type, 0.5)
        self.satisfaction_history.append(satisfaction)

    def _get_avg_satisfaction(self) -> float:
        """Get average user satisfaction"""
        if not self.satisfaction_history:
            return 0.5

        return np.mean(self.satisfaction_history)

    def optimize_clusters(
        self, clusters: List[Dict], recent_feedback: List[Dict]
    ) -> List[Dict]:
        """
        Optimize clusters based on recent feedback

        Args:
            clusters: Current clusters
            recent_feedback: Recent feedback items

        Returns:
            Optimized clusters
        """
        if not clusters or not recent_feedback:
            return clusters

        try:
            # Analyze feedback patterns
            feedback_patterns = self._analyze_feedback_patterns(recent_feedback)

            # Apply optimizations based on patterns
            optimized_clusters = clusters.copy()

            # Check for merge suggestions
            if feedback_patterns.get("should_merge_count", 0) > 0:
                optimized_clusters = self._attempt_merge(optimized_clusters)
                self.cluster_optimization_stats["merge_attempts"] += 1

            # Check for split suggestions
            if feedback_patterns.get("should_split_count", 0) > 0:
                optimized_clusters = self._attempt_split(optimized_clusters)
                self.cluster_optimization_stats["split_attempts"] += 1

            # Check for rebalancing needs
            if feedback_patterns.get("balance_issues", False):
                optimized_clusters = self._attempt_rebalance(optimized_clusters)
                self.cluster_optimization_stats["rebalance_attempts"] += 1

            return optimized_clusters

        except Exception as e:
            logger.error(f"Cluster optimization error: {str(e)}")
            return clusters

    def _analyze_feedback_patterns(self, feedback_list: List[Dict]) -> Dict:
        """Analyze patterns in recent feedback"""
        patterns = {
            "should_merge_count": 0,
            "should_split_count": 0,
            "wrong_cluster_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "balance_issues": False,
        }

        positive_feedback = {"excellent", "good", "relevant", "helpful", "accurate"}
        negative_feedback = {
            "poor",
            "irrelevant",
            "wrong_cluster",
            "confusing",
            "terrible",
        }

        for feedback in feedback_list:
            feedback_type = feedback.get("feedback", "").lower()

            if feedback_type == "should_merge":
                patterns["should_merge_count"] += 1
            elif feedback_type == "should_split":
                patterns["should_split_count"] += 1
            elif feedback_type == "wrong_cluster":
                patterns["wrong_cluster_count"] += 1
            elif feedback_type in positive_feedback:
                patterns["positive_count"] += 1
            elif feedback_type in negative_feedback:
                patterns["negative_count"] += 1

        # Detect balance issues
        if patterns["wrong_cluster_count"] > 2:
            patterns["balance_issues"] = True

        return patterns

    def _attempt_merge(self, clusters: List[Dict]) -> List[Dict]:
        """Attempt to merge similar clusters"""
        if len(clusters) < 2:
            return clusters

        try:
            # Find most similar pair of clusters
            best_similarity = 0
            merge_candidates = None

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    similarity = self._calculate_cluster_similarity(
                        clusters[i], clusters[j]
                    )

                    if similarity > best_similarity and similarity > 0.7:
                        best_similarity = similarity
                        merge_candidates = (i, j)

            # Perform merge if candidates found
            if merge_candidates:
                i, j = merge_candidates
                merged_cluster = self._merge_two_clusters(clusters[i], clusters[j])

                # Create new cluster list
                new_clusters = []
                for idx, cluster in enumerate(clusters):
                    if idx not in merge_candidates:
                        new_clusters.append(cluster)

                new_clusters.append(merged_cluster)

                logger.info(f"Merged clusters {i} and {j} based on feedback")
                return new_clusters

            return clusters

        except Exception as e:
            logger.error(f"Merge attempt error: {str(e)}")
            return clusters

    def _attempt_split(self, clusters: List[Dict]) -> List[Dict]:
        """Attempt to split large or incoherent clusters"""
        try:
            # Find cluster that should be split
            split_candidate = None
            min_coherence = float("inf")

            for i, cluster in enumerate(clusters):
                if cluster["size"] > 6 and cluster["coherence_score"] < min_coherence:
                    min_coherence = cluster["coherence_score"]
                    split_candidate = i

            # Perform split if candidate found
            if split_candidate is not None and min_coherence < 0.5:
                cluster_to_split = clusters[split_candidate]
                split_clusters = self._split_cluster(cluster_to_split)

                # Create new cluster list
                new_clusters = []
                for idx, cluster in enumerate(clusters):
                    if idx != split_candidate:
                        new_clusters.append(cluster)

                new_clusters.extend(split_clusters)

                logger.info(f"Split cluster {split_candidate} based on feedback")
                return new_clusters

            return clusters

        except Exception as e:
            logger.error(f"Split attempt error: {str(e)}")
            return clusters

    def _attempt_rebalance(self, clusters: List[Dict]) -> List[Dict]:
        """Attempt to rebalance cluster sizes"""
        try:
            if len(clusters) < 2:
                return clusters

            # Calculate target size
            total_items = sum(cluster["size"] for cluster in clusters)
            target_size = total_items // len(clusters)

            # Find large and small clusters
            large_clusters = [c for c in clusters if c["size"] > target_size * 1.5]
            small_clusters = [c for c in clusters if c["size"] < target_size * 0.7]

            if not large_clusters or not small_clusters:
                return clusters

            # Move items from large to small clusters (simplified)
            rebalanced_clusters = []

            for cluster in clusters:
                new_cluster = cluster.copy()

                # Simplistic rebalancing - adjust sizes conceptually
                if cluster["size"] > target_size * 1.5:
                    # Large cluster - reduce size slightly
                    adjustment = min(2, cluster["size"] - target_size)
                    new_cluster["size"] -= adjustment
                elif cluster["size"] < target_size * 0.7:
                    # Small cluster - increase size slightly
                    adjustment = min(2, target_size - cluster["size"])
                    new_cluster["size"] += adjustment

                rebalanced_clusters.append(new_cluster)

            logger.info("Rebalanced cluster sizes based on feedback")
            return rebalanced_clusters

        except Exception as e:
            logger.error(f"Rebalance attempt error: {str(e)}")
            return clusters

    def _calculate_cluster_similarity(self, cluster1: Dict, cluster2: Dict) -> float:
        """Calculate similarity between two clusters"""
        try:
            # Compare sources
            sources1 = set(cluster1.get("sources", []))
            sources2 = set(cluster2.get("sources", []))
            source_overlap = (
                len(sources1.intersection(sources2)) / len(sources1.union(sources2))
                if sources1.union(sources2)
                else 0
            )

            # Compare categories
            categories1 = set(cluster1.get("categories", []))
            categories2 = set(cluster2.get("categories", []))
            category_overlap = (
                len(categories1.intersection(categories2))
                / len(categories1.union(categories2))
                if categories1.union(categories2)
                else 0
            )

            # Combine similarities
            return (source_overlap + category_overlap) / 2

        except Exception as e:
            logger.error(f"Similarity calculation error: {str(e)}")
            return 0.0

    def _merge_two_clusters(self, cluster1: Dict, cluster2: Dict) -> Dict:
        """Merge two clusters into one"""
        merged_results = cluster1["results"] + cluster2["results"]

        return {
            "id": max(cluster1["id"], cluster2["id"]),
            "label": f"{cluster1['label']} & {cluster2['label']}",
            "results": merged_results,
            "size": len(merged_results),
            "coherence_score": (
                cluster1["coherence_score"] + cluster2["coherence_score"]
            )
            / 2,
            "diversity_score": max(
                cluster1["diversity_score"], cluster2["diversity_score"]
            ),
            "sources": list(
                set(cluster1.get("sources", []) + cluster2.get("sources", []))
            ),
            "categories": list(
                set(cluster1.get("categories", []) + cluster2.get("categories", []))
            ),
        }

    def _split_cluster(self, cluster: Dict) -> List[Dict]:
        """Split a cluster into two smaller clusters"""
        results = cluster["results"]
        mid_point = len(results) // 2

        cluster_a = {
            "id": cluster["id"],
            "label": f"{cluster['label']} (Part A)",
            "results": results[:mid_point],
            "size": mid_point,
            "coherence_score": cluster["coherence_score"] * 0.9,
            "diversity_score": cluster["diversity_score"] * 0.8,
            "sources": cluster.get("sources", []),
            "categories": cluster.get("categories", []),
        }

        cluster_b = {
            "id": cluster["id"] + 1000,  # Ensure unique ID
            "label": f"{cluster['label']} (Part B)",
            "results": results[mid_point:],
            "size": len(results) - mid_point,
            "coherence_score": cluster["coherence_score"] * 0.9,
            "diversity_score": cluster["diversity_score"] * 0.8,
            "sources": cluster.get("sources", []),
            "categories": cluster.get("categories", []),
        }

        return [cluster_a, cluster_b]

    def get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        """Get recent feedback items"""
        return list(self.feedback_history)[-limit:]

    def get_feedback_count(self) -> int:
        """Get total feedback count"""
        return len(self.feedback_history)

    def get_metrics(self) -> Dict:
        """Get feedback processing metrics"""
        return {
            "total_feedback": len(self.feedback_history),
            "avg_user_satisfaction": self._get_avg_satisfaction(),
            "user_satisfaction_pct": int(self._get_avg_satisfaction() * 100),
            "optimization_stats": self.cluster_optimization_stats.copy(),
            "recent_feedback_count": min(len(self.feedback_history), 10),
        }

    def save_state(self):
        """Save feedback processor state"""
        try:
            state = {
                "feedback_history": list(self.feedback_history),
                "cluster_feedback": dict(self.cluster_feedback),
                "action_rewards": dict(self.action_rewards),
                "optimization_stats": self.cluster_optimization_stats,
                "satisfaction_history": list(self.satisfaction_history),
                "timestamp": datetime.now().isoformat(),
            }

            with open("feedback_processor_state.json", "w") as f:
                json.dump(state, f, indent=2)

            logger.debug("Feedback processor state saved")

        except Exception as e:
            logger.error(f"Save state error: {str(e)}")

    def load_state(self):
        """Load feedback processor state"""
        try:
            with open("feedback_processor_state.json", "r") as f:
                state = json.load(f)

            self.feedback_history = deque(
                state.get("feedback_history", []), maxlen=1000
            )
            self.cluster_feedback = defaultdict(list, state.get("cluster_feedback", {}))
            self.action_rewards = defaultdict(list, state.get("action_rewards", {}))
            self.cluster_optimization_stats = state.get(
                "optimization_stats",
                {
                    "merge_attempts": 0,
                    "split_attempts": 0,
                    "rebalance_attempts": 0,
                    "total_rewards": 0.0,
                    "episodes": 0,
                },
            )
            self.satisfaction_history = deque(
                state.get("satisfaction_history", []), maxlen=100
            )

            logger.info("Feedback processor state loaded")

        except FileNotFoundError:
            logger.info("No previous feedback processor state found")
        except Exception as e:
            logger.error(f"Load state error: {str(e)}")

    def reset(self):
        """Reset feedback processor to initial state"""
        self.feedback_history.clear()
        self.cluster_feedback.clear()
        self.action_rewards.clear()
        self.satisfaction_history.clear()

        self.cluster_optimization_stats = {
            "merge_attempts": 0,
            "split_attempts": 0,
            "rebalance_attempts": 0,
            "total_rewards": 0.0,
            "episodes": 0,
        }

        logger.info("Feedback processor reset")
