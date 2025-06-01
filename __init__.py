"""
Search Simulator Package v2.0
Modular search result simulation for disambiguation research
"""

from .search_simulator import SearchSimulator
from .query_analyzer import QueryAnalyzer
from .user_behavior import UserBehaviorSimulator
from .result_generator import ResultGenerator
from .arabic_support import ArabicResultGenerator

__version__ = "2.0.0"
__author__ = "Search Disambiguation Research Team"

# Main exports
__all__ = [
    'SearchSimulator',
    'QueryAnalyzer', 
    'UserBehaviorSimulator',
    'ResultGenerator',
    'ArabicResultGenerator'
]

# Convenience imports for common use cases
def quick_search(query, language='en', num_results=20):
    """Quick search function for simple testing"""
    simulator = SearchSimulator()
    return simulator.simulate_search(query, language, num_results)

def analyze_query(query):
    """Quick query analysis function"""
    analyzer = QueryAnalyzer()
    return analyzer.analyze(query)

def simulate_user(results, user_type='average'):
    """Quick user behavior simulation"""
    behavior_sim = UserBehaviorSimulator()
    return behavior_sim.simulate_session(results, user_type)