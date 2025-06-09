"""
Dataset Manager Package
Real-world dataset integration for search disambiguation research
"""

import logging
from pathlib import Path
from .core import DatasetManager, DatasetSource
from .database import DatabaseHandler

# Import sources
try:
    from .sources.wikipedia import WikipediaSource
    from .sources.trec import TRECSource  
    from .sources.miracl import MIRACLSource
    from .sources.live_apis import LiveAPISource
except ImportError as e:
    print(f"Warning: Some dataset sources not available: {e}")

__version__ = "1.0.0"

# Main exports
__all__ = [
    "DatasetManager",
    "DatasetSource", 
    "DatabaseHandler",
]
__author__ = "Search Disambiguation Research Team"

logger = logging.getLogger(__name__)


def setup_dataset_manager(
    data_dir: str = "datasets",
    enable_live_apis: bool = True,
    enable_downloads: bool = False,
) -> DatasetManager:
    """
    Setup and initialize the complete dataset manager with all sources.

    Args:
        data_dir: Base directory for storing datasets
        enable_live_apis: Whether to enable live API sources
        enable_downloads: Whether to attempt downloading missing datasets

    Returns:
        Fully configured DatasetManager instance
    """
    logger.info("Setting up comprehensive dataset manager...")

    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    # Initialize database handler
    db_handler = DatabaseHandler(str(data_path / "search_data.db"))

    # Initialize dataset manager
    manager = DatasetManager(data_dir)

    # Register Wikipedia source
    wikipedia_source = WikipediaSource(db_handler)
    manager.register_source("wikipedia", wikipedia_source)

    # Register TREC source
    trec_source = TRECSource(db_handler, str(data_path / "trec"))
    manager.register_source("trec", trec_source)

    # Register MIRACL source
    miracl_source = MIRACLSource(db_handler, str(data_path / "miracl"))
    manager.register_source("miracl", miracl_source)

    # Register live API source if enabled
    if enable_live_apis:
        live_api_source = LiveAPISource(db_handler)
        manager.register_source("live_apis", live_api_source)

    # Initialize all sources
    if manager.initialize():
        logger.info("Dataset manager setup completed successfully")
    else:
        logger.warning("Dataset manager setup completed with some errors")

    return manager


def get_search_results(
    query: str,
    language: str = "en",
    num_results: int = 20,
    manager: DatasetManager = None,
) -> list:
    """
    Convenience function to get search results from real datasets.

    Args:
        query: Search query
        language: Language code ('en' or 'ar')
        num_results: Number of results to return
        manager: Optional DatasetManager instance

    Returns:
        List of real search results
    """
    if manager is None:
        # Create a default manager
        manager = setup_dataset_manager()

    return manager.get_search_results(query, language, num_results)


def get_ambiguous_queries(
    language: str = "en", limit: int = 50, manager: DatasetManager = None
) -> list:
    """
    Convenience function to get ambiguous queries from real datasets.

    Args:
        language: Language code ('en' or 'ar')
        limit: Maximum number of queries to return
        manager: Optional DatasetManager instance

    Returns:
        List of ambiguous queries
    """
    if manager is None:
        manager = setup_dataset_manager()

    return manager.get_ambiguous_queries(language, limit)


def download_datasets(data_dir: str = "datasets", sample_only: bool = True):
    """
    Download available datasets using HuggingFace datasets library.

    Args:
        data_dir: Directory to store datasets
        sample_only: Whether to download only sample data (recommended for testing)
    """
    logger.info("Initiating dataset downloads...")

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    # We'll use direct downloads instead of HuggingFace datasets library
    # to avoid import conflicts with local 'datasets' folder
    print("📦 Using direct download approach (no external dependencies needed)")

    # MIRACL dataset download
    print("🔄 Downloading MIRACL dataset...")
    
    try:
        miracl_dir = data_path / "miracl"
        miracl_source = MIRACLSource(None, str(miracl_dir))
        
        if sample_only:
            print("📥 Downloading sample data for testing...")
            if miracl_source.download_sample_data():
                print("✅ MIRACL sample data downloaded successfully")
                
                # Load the sample data
                db_handler = DatabaseHandler(str(data_path / "search_data.db"))
                miracl_source_with_db = MIRACLSource(db_handler, str(miracl_dir))
                if miracl_source_with_db.load_from_sample():
                    print("✅ MIRACL sample data loaded into database")
                else:
                    print("⚠️  Failed to load sample data into database")
            else:
                print("❌ Failed to download MIRACL sample data")
        else:
            print("📥 Downloading full MIRACL dataset (this may take a while)...")
            if miracl_source.load_data():
                print("✅ MIRACL dataset downloaded and processed successfully")
            else:
                print("❌ Failed to download full MIRACL dataset")
                
    except Exception as e:
        print(f"❌ Error with MIRACL download: {str(e)}")
        print("💡 Try running with sample_only=True for testing")

    # TREC requires manual download
    trec_dir = data_path / "trec"
    trec_dir.mkdir(exist_ok=True)

    print(f"""
    📋 TREC Web Diversity Track datasets require manual download:
    
    1. Visit: https://trec.nist.gov/data/webmain.html
    2. Download topic files (wt09-topics.xml, wt10-topics.xml, etc.)
    3. Download qrels files (wt09.qrels, wt10.qrels, etc.)
    4. Place files in: {trec_dir}
    
    🌍 Wikipedia data will be fetched automatically via API.
    🔗 Live API sources require no setup.
    """)

    return True


def quick_miracl_setup(data_dir: str = "datasets") -> bool:
    """
    Quick setup for MIRACL sample data (recommended for development/testing).
    
    Args:
        data_dir: Directory to store datasets
        
    Returns:
        True if successful, False otherwise
    """
    print("🚀 Quick MIRACL setup for development...")
    
    try:
        data_path = Path(data_dir)
        data_path.mkdir(exist_ok=True)
        
        # Initialize components
        miracl_dir = data_path / "miracl"
        db_handler = DatabaseHandler(str(data_path / "search_data.db"))
        
        # Download sample data
        print("📥 Downloading MIRACL sample data...")
        miracl_source = MIRACLSource(None, str(miracl_dir))
        
        if not miracl_source.download_sample_data():
            print("❌ Failed to download sample data")
            return False
            
        # Load into database
        print("📊 Loading data into database...")
        miracl_source_with_db = MIRACLSource(db_handler, str(miracl_dir))
        
        if not miracl_source_with_db.load_from_sample():
            print("❌ Failed to load sample data into database")
            return False
            
        print("✅ Quick MIRACL setup completed successfully!")
        print(f"📁 Data stored in: {data_path}")
        print("🔍 You can now use the dataset manager for Arabic search queries")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during quick setup: {str(e)}")
        return False


def install_requirements():
    """Print installation instructions for required packages"""
    print("""
    📦 Required packages for dataset manager:
    
    # Core requirements
    pip install datasets
    pip install wikipedia-api
    pip install requests
    
    # Optional for advanced features
    pip install beautifulsoup4
    pip install nltk
    pip install spacy
    
    # For Arabic text processing
    pip install arabic-reshaper
    pip install python-bidi
    
    🚀 Quick install all:
    pip install datasets wikipedia-api requests beautifulsoup4 nltk arabic-reshaper python-bidi
    """)


# Example usage and testing
def test_dataset_manager():
    """Test the dataset manager functionality"""
    print("🧪 Testing Dataset Manager...")

    try:
        # Setup manager
        manager = setup_dataset_manager(enable_live_apis=False)  # Disable for testing

        # Get statistics
        stats = manager.get_statistics()
        print(f"📊 Dataset Statistics: {stats}")

        # Test English search
        print("\n🔍 Testing English search...")
        en_results = manager.get_search_results("python programming", "en", 5)
        print(f"Found {len(en_results)} English results")
        if en_results:
            print(f"📄 Sample result: {en_results[0]['title']}")

        # Test Arabic search
        print("\n🔍 Testing Arabic search...")
        ar_results = manager.get_search_results("محمد صلاح", "ar", 5)
        print(f"Found {len(ar_results)} Arabic results")
        if ar_results:
            print(f"📄 Sample result: {ar_results[0]['title']}")

        # Test ambiguous queries
        print("\n❓ Testing ambiguous queries...")
        en_queries = manager.get_ambiguous_queries("en", 5)
        ar_queries = manager.get_ambiguous_queries("ar", 5)

        print(f"Found {len(en_queries)} English ambiguous queries")
        print(f"Found {len(ar_queries)} Arabic ambiguous queries")

        if en_queries:
            print(f"🔤 Sample English query: {en_queries[0]['query']}")
        if ar_queries:
            print(f"🔤 Sample Arabic query: {ar_queries[0]['query']}")

        print("\n✅ Dataset manager test completed!")
        return manager
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Run test
    test_manager = test_dataset_manager()