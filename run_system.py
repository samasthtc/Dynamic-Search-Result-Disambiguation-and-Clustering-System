#!/usr/bin/env python3
"""
Complete System Runner for DSR-RL
Handles all setup, validation, and execution
OPTIMIZED: Now checks for pre-computed vectors
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if we're in the correct conda environment"""
    try:
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env != 'search-system':
            logger.warning(f"⚠️  Current conda environment: {conda_env}")
            logger.warning("Expected environment: search-system")
            logger.warning("Activate with: conda activate search-system")
            return False
        else:
            logger.info(f"✅ Conda environment: {conda_env}")
            return True
    except Exception as e:
        logger.warning(f"Could not detect conda environment: {e}")
        return True  # Continue anyway

def check_file_structure():
    """Check if all required files are present"""
    required_files = {
        'backend/app.py': 'Main Flask application',
        'backend/clustering_manager.py': 'Clustering algorithms',
        'backend/rl_agent.py': 'Reinforcement learning agent',
        'backend/arabic_processor.py': 'Arabic text processing',
        'backend/search_client.py': 'Search client with real APIs',
        'backend/env_loader.py': 'Environment configuration',
        'frontend/index.html': 'Frontend HTML',
        'frontend/style.css': 'Frontend CSS',
        'frontend/script.js': 'Frontend JavaScript',
        'backend/requirements.txt': 'Python dependencies'
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if not Path(file_path).exists():
            missing_files.append(f"{file_path} ({description})")
    
    if missing_files:
        logger.error("❌ Missing required files:")
        for file in missing_files:
            logger.error(f"   - {file}")
        return False
    else:
        logger.info("✅ All required files present")
        return True

def check_dependencies():
    """Check if all required Python packages are installed"""
    required_packages = [
        'flask', 'flask_cors', 'numpy', 'pandas', 'sklearn',
        'sentence_transformers', 'hdbscan', 'bertopic', 'datasets',
        'pyarabic', 'wikipediaapi', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("❌ Missing required packages:")
        for package in missing_packages:
            logger.error(f"   - {package}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    else:
        logger.info("✅ All required packages installed")
        return True

def check_api_configuration():
    """Check API configuration"""
    google_api_key = os.getenv('GOOGLE_API_KEY')
    google_cse_id = os.getenv('GOOGLE_CSE_ID')
    
    if google_api_key and google_cse_id:
        logger.info("✅ Google Custom Search API configured")
        return True
    else:
        logger.warning("⚠️  Google Custom Search API not configured")
        logger.warning("The system will work with limited search functionality")
        logger.warning("Set up with: python setup.py")
        return False

def check_optimization_status():
    """NEW: Check if datasets and vectors are optimized"""
    logger.info("🔍 Checking optimization status...")
    
    cache_files = {
        'miracl_arabic_cache.pkl': 'MIRACL Arabic corpus',
        'miracl_arabic_vectors.pkl': 'Pre-computed Arabic vectors',
        'trec_web_diversity_cache.pkl': 'TREC Web Diversity corpus'
    }
    
    optimization_status = {}
    
    for cache_file, description in cache_files.items():
        if Path(cache_file).exists():
            file_size = Path(cache_file).stat().st_size / (1024 * 1024)  # MB
            logger.info(f"✅ {description}: Cached ({file_size:.1f} MB)")
            optimization_status[cache_file] = True
        else:
            logger.warning(f"⚠️  {description}: Not cached")
            optimization_status[cache_file] = False
    
    # Check if Arabic search is optimized
    if optimization_status.get('miracl_arabic_vectors.pkl', False):
        logger.info("🚀 Arabic search: OPTIMIZED (fast)")
        arabic_optimized = True
    else:
        logger.warning("⚠️  Arabic search: NOT OPTIMIZED (will be slow)")
        logger.warning("Run 'python setup.py' for optimization")
        arabic_optimized = False
    
    return {
        'arabic_optimized': arabic_optimized,
        'corpus_cached': optimization_status.get('miracl_arabic_cache.pkl', False),
        'trec_cached': optimization_status.get('trec_web_diversity_cache.pkl', False),
        'vectors_cached': optimization_status.get('miracl_arabic_vectors.pkl', False)
    }

def run_performance_tests():
    """NEW: Test search performance"""
    logger.info("🧪 Running performance tests...")
    
    try:
        sys.path.append('backend')
        from search_client import SearchClient
        
        # Initialize search client
        search_client = SearchClient()
        stats = search_client.get_dataset_statistics()
        
        # Test Arabic search performance if optimized
        if stats.get('miracl_vectors_precomputed', False):
            logger.info("⚡ Testing optimized Arabic search performance...")
            
            test_queries = ["تكنولوجيا", "التعليم", "الذكاء"]
            total_time = 0
            
            for query in test_queries:
                start_time = time.time()
                results = search_client.search_specific_dataset(query, "miracl_arabic", 5)
                end_time = time.time()
                query_time = end_time - start_time
                total_time += query_time
                logger.info(f"   - '{query}': {len(results)} results in {query_time:.3f}s")
            
            avg_time = total_time / len(test_queries)
            if avg_time < 0.1:
                logger.info(f"🚀 Arabic search performance: EXCELLENT ({avg_time:.3f}s avg)")
            elif avg_time < 0.5:
                logger.info(f"⚡ Arabic search performance: GOOD ({avg_time:.3f}s avg)")
            else:
                logger.warning(f"⚠️  Arabic search performance: SLOW ({avg_time:.3f}s avg)")
        else:
            logger.warning("⚠️  Arabic search not optimized, skipping performance test")
        
        # Test English search with TREC
        logger.info("🔍 Testing English ambiguous search...")
        start_time = time.time()
        trec_results = search_client.search_specific_dataset("Jackson", "trec", 5)
        end_time = time.time()
        logger.info(f"   - TREC search: {len(trec_results)} results in {end_time-start_time:.3f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance tests failed: {e}")
        return False

def run_system_tests():
    """Run basic system tests"""
    logger.info("🧪 Running system tests...")
    
    try:
        # Test imports
        sys.path.append('backend')
        
        from clustering_manager import ClusteringManager
        from rl_agent import RLAgent
        from arabic_processor import ArabicProcessor
        from search_client import SearchClient
        from env_loader import load_environment_variables
        
        # Test component initialization
        arabic_processor = ArabicProcessor()
        test_text = "هذا نص تجريبي"
        processed = arabic_processor.preprocess_text(test_text)
        logger.info(f"✅ Arabic processing test: '{test_text}' → '{processed}'")
        
        clustering_manager = ClusteringManager()
        logger.info("✅ Clustering manager initialized")
        
        rl_agent = RLAgent()
        logger.info("✅ RL agent initialized")
        
        search_client = SearchClient()
        logger.info("✅ Search client initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        return False

def suggest_optimization():
    """Suggest optimization steps if needed"""
    optimization = check_optimization_status()
    
    if not optimization['arabic_optimized']:
        print("\n" + "🚀 OPTIMIZATION RECOMMENDATION" + "\n" + "=" * 40)
        print("For faster Arabic search performance, run:")
        print("   python setup.py")
        print("\nThis will:")
        print("   - Download MIRACL-Arabic corpus")
        print("   - Pre-compute TF-IDF vectors")
        print("   - Cache everything for instant search")
        print("\nFirst-time setup takes 5-10 minutes but makes searches 100x faster!")
        
        print("\nContinue anyway? (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("👋 Run setup first, then come back!")
            return False
    
    return True

def start_flask_server():
    """Start the Flask server"""
    logger.info("🚀 Starting DSR-RL Flask server...")
    
    try:
        os.chdir('backend')
        
        # Set Flask environment variables
        os.environ['FLASK_APP'] = 'app.py'
        os.environ['FLASK_ENV'] = 'development'
        
        # Start Flask server
        logger.info("🌐 Server starting on http://localhost:5000")
        logger.info("📱 Open this URL in your web browser")
        logger.info("🛑 Press Ctrl+C to stop the server")
        
        # Run Flask
        subprocess.run([sys.executable, 'app.py'])
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")

def main():
    """Main system runner"""
    print("🔬 DSR-RL System Runner - OPTIMIZED VERSION")
    print("Dynamic Search Result Disambiguation and Clustering via Reinforcement Learning")
    print("=" * 80)
    
    # Pre-flight checks
    logger.info("🔍 Running pre-flight checks...")
    
    checks = [
        ("Environment", check_environment),
        ("File Structure", check_file_structure),
        ("Dependencies", check_dependencies),
        ("API Configuration", check_api_configuration),
        ("System Tests", run_system_tests)
    ]
    
    failed_checks = []
    for check_name, check_func in checks:
        logger.info(f"\n📋 Checking: {check_name}")
        if not check_func():
            failed_checks.append(check_name)
    
    # Check optimization status
    logger.info(f"\n📋 Checking: Optimization Status")
    optimization = check_optimization_status()
    
    # Run performance tests if possible
    logger.info(f"\n📋 Checking: Performance")
    performance_ok = run_performance_tests()
    
    if failed_checks:
        logger.error(f"\n❌ Failed checks: {', '.join(failed_checks)}")
        logger.error("Please fix the issues above before running the system")
        logger.error("Run 'python setup.py' for automated setup")
        return False
    
    logger.info("\n✅ All critical checks passed!")
    
    # Show optimization status
    print("\n" + "🚀 OPTIMIZATION STATUS" + "\n" + "=" * 30)
    if optimization['arabic_optimized']:
        print("✅ Arabic Search: OPTIMIZED (lightning fast)")
    else:
        print("⚠️  Arabic Search: NOT OPTIMIZED (will be slow)")
    
    if optimization['vectors_cached']:
        print("✅ TF-IDF Vectors: PRE-COMPUTED")
    else:
        print("⚠️  TF-IDF Vectors: WILL COMPUTE ON DEMAND")
    
    if optimization['corpus_cached']:
        print("✅ MIRACL Corpus: CACHED")
    else:
        print("⚠️  MIRACL Corpus: WILL DOWNLOAD ON DEMAND")
    
    if optimization['trec_cached']:
        print("✅ TREC Dataset: CACHED")
    else:
        print("⚠️  TREC Dataset: WILL USE FALLBACK")
    
    # Suggest optimization if needed
    if not suggest_optimization():
        return False
    
    # Start the system
    logger.info("\n" + "=" * 80)
    print("\n🎯 READY TO START!")
    print("Features available:")
    print("   🔍 Real-time search disambiguation")
    print("   🤖 Reinforcement learning optimization")
    print("   🌐 Multi-language support (English/Arabic)")
    print("   📊 Advanced clustering algorithms")
    print("   ⚡ Optimized performance" + (" (ENABLED)" if optimization['arabic_optimized'] else " (run setup.py)"))
    
    time.sleep(1)
    start_flask_server()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
        sys.exit(0)