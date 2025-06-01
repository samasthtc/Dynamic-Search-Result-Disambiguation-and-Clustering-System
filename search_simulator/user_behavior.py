"""
User behavior simulation module
Models realistic user interactions with search results
"""

import random
import numpy as np
from typing import List, Dict, Any
import logging

from .data_templates import USER_PROFILES, INTENT_MODIFIERS

logger = logging.getLogger(__name__)


class UserBehaviorSimulator:
    """
    Simulates realistic user behavior patterns with search results.
    Models different user types and search intents.
    """
    
    def __init__(self):
        self.position_bias_factor = 0.15
        self.authority_bias_factor = 0.1
        
    def simulate_session(self, results: List[Dict], user_type: str = 'average',
                        intent: str = 'informational') -> Dict[str, Any]:
        """
        Simulate a complete user search session.
        
        Args:
            results: List of search results
            user_type: Type of user ('novice', 'average', 'expert', 'researcher')
            intent: Search intent ('informational', 'navigational', 'transactional', 'commercial')
            
        Returns:
            Complete session analysis
        """
        if user_type not in USER_PROFILES:
            user_type = 'average'
        
        if intent not in INTENT_MODIFIERS:
            intent = 'informational'
        
        # Get user profile and intent modifiers
        profile = USER_PROFILES[user_type]
        intent_mod = INTENT_MODIFIERS[intent]
        
        # Adjust profile based on intent
        adjusted_profile = self._adjust_profile_for_intent(profile, intent_mod)
        
        # Simulate session
        session_data = self._simulate_user_interactions(results, adjusted_profile)
        
        # Analyze patterns
        patterns = self._analyze_behavioral_patterns(session_data)
        
        # Calculate session metrics
        metrics = self._calculate_session_metrics(session_data, adjusted_profile)
        
        return {
            'user_type': user_type,
            'intent': intent,
            'interactions': session_data,
            'patterns': patterns,
            'metrics': metrics,
            'session_summary': self._create_session_summary(session_data, metrics)
        }
    
    def _adjust_profile_for_intent(self, profile: Dict, intent_mod: Dict) -> Dict:
        """Adjust user profile based on search intent"""
        adjusted = profile.copy()
        
        # Apply intent modifiers
        adjusted['attention_span'] *= intent_mod['patience_factor']
        adjusted['results_examined'] = min(1.0, adjusted['results_examined'] * intent_mod['depth_factor'])
        
        # Intent-specific adjustments
        if intent_mod == INTENT_MODIFIERS['navigational']:
            adjusted['click_threshold'] *= 0.8  # More likely to click quickly
        elif intent_mod == INTENT_MODIFIERS['transactional']:
            adjusted['brand_preference'] *= 1.2  # More brand conscious
        
        return adjusted
    
    def _simulate_user_interactions(self, results: List[Dict], 
                                  profile: Dict) -> List[Dict]:
        """Simulate user interactions with each result"""
        interactions = []
        
        # Determine how many results user will examine
        num_to_examine = max(1, int(len(results) * profile['results_examined']))
        examined_results = results[:num_to_examine]
        
        total_time = 0
        
        for i, result in enumerate(examined_results):
            interaction = self._simulate_single_interaction(result, i, profile)
            interaction['cumulative_time'] = total_time + interaction['scan_time']
            total_time = interaction['cumulative_time']
            
            # Add dwell time if clicked
            if interaction['clicked']:
                dwell_time = self._simulate_dwell_time(result, profile)
                interaction['dwell_time'] = dwell_time
                total_time += dwell_time
                interaction['cumulative_time'] = total_time
            
            interactions.append(interaction)
        
        return interactions
    
    def _simulate_single_interaction(self, result: Dict, position: int, 
                                   profile: Dict) -> Dict:
        """Simulate interaction with a single result"""
        # Calculate scan time
        base_scan_time = profile['attention_span']
        scan_variance = random.normalvariate(0, base_scan_time * 0.3)
        scan_time = max(0.5, base_scan_time + scan_variance)
        
        # Calculate click probability
        click_prob = self._calculate_click_probability(result, position, profile)
        
        # Determine if clicked
        clicked = random.random() < click_prob
        
        interaction = {
            'result_id': result['id'],
            'position': position + 1,
            'scan_time': scan_time,
            'click_probability': click_prob,
            'clicked': clicked,
            'result_title': result['title'],
            'result_domain': result['domain'],
            'result_score': result.get('final_score', 0.5)
        }
        
        # If clicked, determine satisfaction
        if clicked:
            satisfaction_prob = self._calculate_satisfaction_probability(result, profile)
            interaction['satisfied'] = random.random() < satisfaction_prob
            interaction['satisfaction_probability'] = satisfaction_prob
        
        return interaction
    
    def _calculate_click_probability(self, result: Dict, position: int, 
                                   profile: Dict) -> float:
        """Calculate probability of clicking a result"""
        # Base click probability from result
        base_prob = result.get('click_probability', 0.1)
        
        # Position bias (higher positions get more clicks)
        position_bias = 1.0 / (1.0 + position * self.position_bias_factor)
        
        # Authority bias
        authority_bias = 1.0
        if result.get('authority_score', 0.5) > 0.8:
            authority_bias += profile['brand_preference'] * self.authority_bias_factor
        
        # Quality threshold
        quality_factor = 1.0
        if result.get('final_score', 0.5) < profile['click_threshold']:
            quality_factor = 0.5  # Less likely to click low-quality results
        
        # Combine factors
        click_prob = base_prob * position_bias * authority_bias * quality_factor
        
        return max(0.01, min(0.9, click_prob))
    
    def _simulate_dwell_time(self, result: Dict, profile: Dict) -> float:
        """Simulate time spent on clicked page"""
        # Base dwell time from result prediction
        predicted_dwell = result.get('dwell_time_prediction', 60)
        
        # Adjust based on user profile
        attention_factor = profile['attention_span'] / 3.0  # Normalize to ~1.0
        quality_factor = result.get('final_score', 0.5) + 0.5  # 0.5-1.5 range
        
        # Calculate actual dwell time with variance
        base_dwell = predicted_dwell * attention_factor * quality_factor
        dwell_variance = random.normalvariate(0, base_dwell * 0.4)
        actual_dwell = max(5, base_dwell + dwell_variance)
        
        return actual_dwell
    
    def _calculate_satisfaction_probability(self, result: Dict, profile: Dict) -> float:
        """Calculate probability of user being satisfied"""
        # Base satisfaction from result
        base_satisfaction = result.get('satisfaction_prediction', 0.5)
        
        # Adjust based on user expectations
        threshold_factor = 1.0
        if base_satisfaction < profile['satisfaction_threshold']:
            threshold_factor = 0.7  # Lower satisfaction if below threshold
        
        # Quality factor
        quality_factor = result.get('final_score', 0.5)
        
        satisfaction_prob = base_satisfaction * threshold_factor * quality_factor
        return max(0.1, min(1.0, satisfaction_prob))
    
    def _analyze_behavioral_patterns(self, interactions: List[Dict]) -> List[str]:
        """Analyze and identify behavioral patterns"""
        patterns = []
        
        if not interactions:
            return patterns
        
        # Calculate metrics for pattern detection
        total_interactions = len(interactions)
        clicked_interactions = [i for i in interactions if i['clicked']]
        total_clicks = len(clicked_interactions)
        
        # Average scan time
        avg_scan_time = sum(i['scan_time'] for i in interactions) / total_interactions
        
        # Pattern detection
        
        # Quick scanner pattern
        if avg_scan_time < 2.0:
            patterns.append('quick_scanner')
        
        # Deep reader pattern
        if any(i.get('dwell_time', 0) > 120 for i in clicked_interactions):
            patterns.append('deep_reader')
        
        # Position bias pattern
        if total_clicks > 0:
            early_clicks = sum(1 for i in clicked_interactions if i['position'] <= 3)
            if early_clicks / total_clicks > 0.8:
                patterns.append('position_biased')
        
        # Brand conscious pattern
        high_authority_clicks = sum(1 for i in clicked_interactions 
                                   if i.get('result_score', 0) > 0.8)
        if high_authority_clicks > 0:
            patterns.append('brand_conscious')
        
        # Comparison shopper pattern
        if total_clicks > 2:
            patterns.append('comparison_shopper')
        
        # Thorough researcher pattern
        if total_interactions > len(interactions) * 0.8 and avg_scan_time > 3.0:
            patterns.append('thorough_researcher')
        
        # Impatient user pattern
        if avg_scan_time < 1.5 and total_clicks <= 1:
            patterns.append('impatient_user')
        
        return patterns
    
    def _calculate_session_metrics(self, interactions: List[Dict], 
                                 profile: Dict) -> Dict[str, Any]:
        """Calculate comprehensive session metrics"""
        if not interactions:
            return {'error': 'No interactions to analyze'}
        
        # Basic counts
        total_results_examined = len(interactions)
        total_clicks = sum(1 for i in interactions if i['clicked'])
        satisfied_clicks = sum(1 for i in interactions 
                              if i['clicked'] and i.get('satisfied', False))
        
        # Time metrics
        total_scan_time = sum(i['scan_time'] for i in interactions)
        total_dwell_time = sum(i.get('dwell_time', 0) for i in interactions)
        total_session_time = total_scan_time + total_dwell_time
        
        # Calculate rates
        click_through_rate = total_clicks / total_results_examined if total_results_examined > 0 else 0
        satisfaction_rate = satisfied_clicks / total_clicks if total_clicks > 0 else 0
        
        # Position analysis
        click_positions = [i['position'] for i in interactions if i['clicked']]
        avg_click_position = sum(click_positions) / len(click_positions) if click_positions else 0
        
        # Quality analysis
        clicked_scores = [i['result_score'] for i in interactions if i['clicked']]
        avg_clicked_quality = sum(clicked_scores) / len(clicked_scores) if clicked_scores else 0
        
        # Engagement depth
        if total_clicks > 0:
            avg_dwell_time = total_dwell_time / total_clicks
            engagement_depth = 'high' if avg_dwell_time > 120 else 'medium' if avg_dwell_time > 60 else 'low'
        else:
            avg_dwell_time = 0
            engagement_depth = 'none'
        
        # Overall session satisfaction
        if satisfied_clicks > 0:
            session_satisfaction = 'high'
        elif total_clicks > 0:
            session_satisfaction = 'medium'
        else:
            session_satisfaction = 'low'
        
        return {
            'results_examined': total_results_examined,
            'clicks': total_clicks,
            'satisfied_clicks': satisfied_clicks,
            'click_through_rate': click_through_rate,
            'satisfaction_rate': satisfaction_rate,
            'total_scan_time': total_scan_time,
            'total_dwell_time': total_dwell_time,
            'total_session_time': total_session_time,
            'avg_dwell_time': avg_dwell_time,
            'avg_click_position': avg_click_position,
            'avg_clicked_quality': avg_clicked_quality,
            'engagement_depth': engagement_depth,
            'session_satisfaction': session_satisfaction
        }
    
    def _create_session_summary(self, interactions: List[Dict], 
                              metrics: Dict) -> Dict[str, Any]:
        """Create a human-readable session summary"""
        if 'error' in metrics:
            return {'summary': 'No valid session data'}
        
        # Determine user behavior type
        ctr = metrics['click_through_rate']
        satisfaction = metrics['satisfaction_rate']
        engagement = metrics['engagement_depth']
        
        if ctr > 0.3 and satisfaction > 0.7:
            behavior_type = 'highly_engaged'
        elif ctr > 0.15 and satisfaction > 0.5:
            behavior_type = 'moderately_engaged'
        elif ctr > 0.05:
            behavior_type = 'browsing'
        else:
            behavior_type = 'scanning_only'
        
        # Generate insights
        insights = []
        
        if metrics['avg_click_position'] <= 2:
            insights.append('User shows strong position bias, clicking mainly on top results')
        
        if metrics['avg_clicked_quality'] > 0.8:
            insights.append('User tends to click on high-quality results')
        
        if engagement == 'high':
            insights.append('User shows deep engagement with content')
        elif engagement == 'low':
            insights.append('User has short attention span')
        
        if metrics['clicks'] > 3:
            insights.append('User exhibits comparison shopping behavior')
        
        return {
            'behavior_type': behavior_type,
            'primary_insight': insights[0] if insights else 'Standard browsing behavior',
            'all_insights': insights,
            'engagement_level': engagement,
            'success_indicator': metrics['session_satisfaction']
        }
    
    def analyze_intent_signals(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Analyze search intent signals from user behavior"""
        if not interactions:
            return {'confidence': 0.0}
        
        clicked_interactions = [i for i in interactions if i['clicked']]
        
        if not clicked_interactions:
            return {'primary_intent': 'unknown', 'confidence': 0.0}
        
        # Analyze interaction patterns
        intent_signals = {
            'informational': 0.0,
            'navigational': 0.0,
            'transactional': 0.0,
            'commercial': 0.0
        }
        
        # Quick clicking suggests navigational intent
        quick_clicks = sum(1 for i in clicked_interactions if i['scan_time'] < 2.0)
        if quick_clicks > 0:
            intent_signals['navigational'] = quick_clicks / len(clicked_interactions)
        
        # Long dwell times suggest informational intent
        long_dwells = sum(1 for i in clicked_interactions if i.get('dwell_time', 0) > 120)
        if long_dwells > 0:
            intent_signals['informational'] = long_dwells / len(clicked_interactions)
        
        # Multiple clicks suggest commercial/comparison intent
        if len(clicked_interactions) > 2:
            intent_signals['commercial'] = min(1.0, len(clicked_interactions) / 5)
        
        # High-authority preference suggests transactional intent
        high_auth_clicks = sum(1 for i in clicked_interactions if i.get('result_score', 0) > 0.8)
        if high_auth_clicks > 0:
            intent_signals['transactional'] = high_auth_clicks / len(clicked_interactions)
        
        # Find primary intent
        primary_intent = max(intent_signals.items(), key=lambda x: x[1])
        
        return {
            'primary_intent': primary_intent[0],
            'confidence': primary_intent[1],
            'intent_distribution': intent_signals,
            'analysis_basis': f"{len(clicked_interactions)} clicks analyzed"
        }
