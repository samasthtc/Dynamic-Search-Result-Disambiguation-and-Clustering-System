#!/usr/bin/env python3
"""
Setup script for DSR-RL system
Handles Google Custom Search API setup and dataset downloads
OPTIMIZED: Now pre-computes Arabic search vectors during setup
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_google_custom_search():
    """Setup Google Custom Search API credentials"""
    print("\n" + "=" * 60)
    print("GOOGLE CUSTOM SEARCH API SETUP")
    print("=" * 60)

    print(
        "\nTo use real Google search results, you need to set up Google Custom Search API:"
    )
    print("\n1. Go to: https://console.developers.google.com/")
    print("2. Create a new project or select existing one")
    print("3. Enable 'Custom Search API'")
    print("4. Create credentials (API Key)")
    print("5. Go to: https://cse.google.com/cse/")
    print("6. Create a new Custom Search Engine")
    print("7. Get your Search Engine ID")

    print("\nDo you want to set up the API credentials now? (y/n): ", end="")
    response = input().strip().lower()

    if response == "y":
        print("\nEnter your Google API Key: ", end="")
        api_key = input().strip()

        print("Enter your Custom Search Engine ID: ", end="")
        cse_id = input().strip()

        if api_key and cse_id:
            # Create .env file
            env_content = f"""# Google Custom Search API Configuration
GOOGLE_API_KEY={api_key}
GOOGLE_CSE_ID={cse_id}
"""

            with open(".env", "w") as f:
                f.write(env_content)

            # Also set environment variables for current session
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GOOGLE_CSE_ID"] = cse_id

            print("\n✅ Google Custom Search API configured successfully!")
            print("Credentials saved to .env file")

            return True
        else:
            print("\n❌ Invalid credentials provided")
            return False
    else:
        print("\n⚠️  Skipping Google Custom Search setup")
        print("The system will work with Wikipedia and datasets only")
        return False


def install_optional_packages():
    """Install optional packages for better dataset access"""
    print("\n" + "=" * 60)
    print("OPTIONAL PACKAGES INSTALLATION")
    print("=" * 60)

    print("\nInstalling optional packages for better dataset access...")

    optional_packages = [
        "ir-datasets",  # For TREC dataset access
        "python-dotenv",  # For .env file support
    ]

    for package in optional_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package} (optional)")


def download_nltk_data():
    """Download required NLTK data"""
    print("\n" + "=" * 60)
    print("NLTK DATA DOWNLOAD")
    print("=" * 60)

    try:
        import nltk

        print("Downloading NLTK data...")

        nltk_data = ["punkt", "stopwords", "wordnet", "omw-1.4"]

        for data in nltk_data:
            try:
                nltk.download(data, quiet=True)
                print(f"✅ Downloaded {data}")
            except Exception as e:
                print(f"⚠️  Failed to download {data}: {e}")

    except ImportError:
        print("⚠️  NLTK not installed, skipping NLTK data download")


def test_imports():
    """Test that all required packages can be imported"""
    print("\n" + "=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)

    required_packages = [
        ("flask", "Flask"),
        ("sklearn", "scikit-learn"),
        ("sentence_transformers", "sentence-transformers"),
        ("hdbscan", "hdbscan"),
        ("bertopic", "bertopic"),
        ("datasets", "datasets"),
        ("pyarabic.araby", "pyarabic"),
        ("wikipediaapi", "wikipedia-api"),
        ("requests", "requests"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
    ]

    failed_imports = []

    for package, pip_name in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (install with: pip install {pip_name})")
            failed_imports.append(pip_name)

    if failed_imports:
        print(f"\n⚠️  Missing packages: {', '.join(failed_imports)}")
        print("Install them with: pip install " + " ".join(failed_imports))
        return False
    else:
        print("\n✅ All required packages are available!")
        return True


def create_directory_structure():
    """Create necessary directory structure"""
    print("\n" + "=" * 60)
    print("CREATING DIRECTORY STRUCTURE")
    print("=" * 60)

    directories = ["backend", "frontend", "data", "logs", "cache"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_env_template():
    """Create template .env file if it doesn't exist"""
    if not os.path.exists(".env"):
        env_template = """# Google Custom Search API Configuration
# Get these from: https://console.developers.google.com/ and https://cse.google.com/
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here

# Optional: Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True
"""

        with open(".env.template", "w") as f:
            f.write(env_template)

        print("✅ Created .env.template file")
        print("Copy it to .env and add your API credentials")


def setup_datasets_and_vectors():
    """NEW: Download datasets and pre-compute vectors for fast search"""
    print("\n" + "=" * 60)
    print("DATASET SETUP AND OPTIMIZATION")
    print("=" * 60)
    
    print("\nThis step will:")
    print("1. Download MIRACL-Arabic corpus (if not cached)")
    print("2. Download TREC Web Diversity dataset (if not cached)")
    print("3. Pre-compute TF-IDF vectors for fast Arabic search")
    print("4. Cache everything for instant startup")
    
    print("\n⚠️  This may take 5-10 minutes on first run but will make searches much faster!")
    print("Continue with dataset setup? (y/n): ", end="")
    response = input().strip().lower()
    
    if response != "y":
        print("⚠️  Skipping dataset optimization. Search will be slower.")
        return False
    
    try:
        # Import our modules
        sys.path.append('backend')
        from search_client import SearchClient
        
        print("\n🔄 Initializing search client and downloading datasets...")
        
        # Initialize search client - this will trigger dataset downloads
        search_client = SearchClient()
        
        print("\n🔄 Pre-computing Arabic search vectors...")
        
        # Force vector pre-computation
        vector_success = search_client.precompute_vectors_for_setup()
        
        if vector_success:
            print("✅ Arabic search vectors pre-computed and cached!")
        else:
            print("⚠️  Arabic vector pre-computation failed, search will be slower")
        
        # Get dataset statistics
        stats = search_client.get_dataset_statistics()
        print(f"\n📊 Dataset statistics:")
        print(f"   - MIRACL Arabic docs: {stats['miracl_arabic_docs']}")
        print(f"   - TREC docs: {stats['trec_docs']}")
        print(f"   - Arabic vectors cached: {stats['optimization_status']['arabic_vectors_cached']}")
        print(f"   - Google API configured: {stats['google_api_configured']}")
        
        print("\n✅ Dataset setup and optimization completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset setup failed: {e}")
        print("The system will still work, but search may be slower.")
        return False


def run_system_tests():
    """Run comprehensive system tests"""
    print("\n" + "=" * 60)
    print("SYSTEM TESTS")
    print("=" * 60)
    
    try:
        sys.path.append('backend')
        
        print("🧪 Testing component imports...")
        
        from clustering_manager import ClusteringManager
        from rl_agent import RLAgent
        from arabic_processor import ArabicProcessor
        from search_client import SearchClient
        
        print("✅ All components imported successfully")
        
        print("\n🧪 Testing Arabic processing...")
        arabic_processor = ArabicProcessor()
        test_text = "هذا نص تجريبي للاختبار"
        processed = arabic_processor.preprocess_text(test_text)
        print(f"✅ Arabic test: '{test_text}' → '{processed}'")
        
        print("\n🧪 Testing search client...")
        search_client = SearchClient()
        stats = search_client.get_dataset_statistics()
        print(f"✅ Search client initialized with {stats['miracl_arabic_docs']} Arabic docs")
        
        print("\n🧪 Testing clustering manager...")
        clustering_manager = ClusteringManager()
        print("✅ Clustering manager initialized")
        
        print("\n🧪 Testing RL agent...")
        rl_agent = RLAgent()
        print("✅ RL agent initialized")
        
        # Test a simple search if vectors are available
        if stats.get('miracl_vectors_precomputed', False):
            print("\n🧪 Testing optimized Arabic search...")
            start_time = time.time()
            results = search_client.search_specific_dataset("تكنولوجيا", "miracl_arabic", 5)
            end_time = time.time()
            print(f"✅ Arabic search test: {len(results)} results in {end_time-start_time:.2f}s")
        
        print("\n✅ All system tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ System tests failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🔬 DSR-RL System Setup - OPTIMIZED VERSION")
    print(
        "Dynamic Search Result Disambiguation and Clustering via Reinforcement Learning"
    )
    print("Based on research by Birzeit University")
    print("\n🚀 NEW: Includes Arabic search optimization!")

    # Test imports first
    if not test_imports():
        print("\n❌ Some required packages are missing.")
        print("Please install them first using: pip install -r requirements.txt")
        return False

    # Create directory structure
    create_directory_structure()

    # Download NLTK data
    download_nltk_data()

    # Install optional packages
    install_optional_packages()

    # Create env template
    create_env_template()

    # Setup Google Custom Search API
    google_setup = setup_google_custom_search()

    # NEW: Setup datasets and pre-compute vectors
    dataset_setup = setup_datasets_and_vectors()

    # Run system tests
    test_success = run_system_tests()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)

    print("\n✅ DSR-RL system setup completed!")

    if dataset_setup:
        print("\n🚀 Arabic search optimization: ENABLED")
        print("   - Arabic queries will be lightning fast!")
    else:
        print("\n⚠️  Arabic search optimization: DISABLED")
        print("   - Arabic queries will be slower on first run")

    if google_setup:
        print("🌐 Google Custom Search: ENABLED")
    else:
        print("⚠️  Google Custom Search: DISABLED")

    if test_success:
        print("🧪 System tests: PASSED")
    else:
        print("⚠️  System tests: FAILED")

    print("\n" + "=" * 40)
    print("READY TO RUN!")
    print("=" * 40)

    print("\n🚀 Start the system with:")
    print("   python run_system.py")
    print("\n📖 Or manually:")
    print("   1. cd backend")
    print("   2. python app.py")
    print("   3. Open http://localhost:5000")

    print("\n💡 Tips:")
    print("   - Arabic search is now pre-optimized for speed")
    print("   - Try queries like 'Jackson', 'Apple', 'Python' for English")
    print("   - Try Arabic queries for fast multilingual search")
    print("   - Check logs for detailed performance metrics")

    return True


if __name__ == "__main__":
    main()