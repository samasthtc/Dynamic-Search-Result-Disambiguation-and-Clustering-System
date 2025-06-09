#!/usr/bin/env python3
"""
Setup Script for Dynamic Search Disambiguation System
UPDATED: Now collects REAL data from Wikipedia and ArXiv (~10 results per term)
"""

import os
import sys
import subprocess
import logging
import requests
import time
import json
import sqlite3
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataSetup:
    """Handles real data collection and system setup"""
    
    def __init__(self):
        self.data_dir = Path("datasets")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "real_search_data.db"
        self.sentence_model = None
        
        # Curated ambiguous terms for real data (limited to ~10 results per term)
        self.ambiguous_terms = {
            "en": [
                "python",    # Programming vs snake
                "apple",     # Company vs fruit  
                "java",      # Programming vs island
                "mercury",   # Planet vs element
                "mars",      # Planet vs company
                "amazon",    # Company vs river
                "oracle",    # Company vs ancient
                "jackson",   # Person vs place
                "paris"      # City vs person
            ],
            "ar": [
                "عين",       # Eye vs spring
                "بنك",       # Bank vs river bank
                "ورد",       # Rose vs mentioned
                "سلم",       # Peace vs ladder
                "نور"        # Light vs name
            ]
        }

def install_requirements():
    """Install required packages"""
    print("📦 Installing packages for real data collection...")
    
    packages = [
        "sentence-transformers",
        "requests",
        "flask", 
        "flask-cors",
        "numpy",
        "pandas",
        "scikit-learn"
    ]
    
    optional_packages = [
        "hdbscan",
        "wikipedia-api",
        "arabic-reshaper", 
        "python-bidi"
    ]

    # Install core packages
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    # Install optional packages
    for package in optional_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Optional package {package} failed - continuing")

def setup_directories():
    """Create necessary directories"""
    dirs = ["datasets", "datasets/miracl", "datasets/trec", "static"]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def collect_real_data():
    """Collect real data from Wikipedia and ArXiv"""
    print("🌐 Collecting REAL data from Wikipedia and ArXiv...")
    print("This may take a few minutes...")
    
    setup = RealDataSetup()
    
    # Initialize database
    setup.init_database()
    
    # Load sentence transformer
    try:
        setup.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer loaded")
    except Exception as e:
        print(f"⚠️  Sentence transformer not available: {e}")
        setup.sentence_model = None
    
    total_collected = 0
    
    # Collect Wikipedia data for each term
    for language in ["en", "ar"]:
        for term in setup.ambiguous_terms[language]:
            try:
                print(f"  🔍 Processing: {term} ({language})")
                
                # Get Wikipedia disambiguation page
                count = setup.fetch_wikipedia_data(term, language)
                total_collected += count
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"  ⚠️  Error with {term}: {e}")
                continue
    
    # Collect some ArXiv data for technical terms
    technical_terms = ["python", "java", "machine learning"]
    for term in technical_terms[:2]:  # Limit to 2 terms
        try:
            print(f"  🔬 ArXiv: {term}")
            count = setup.fetch_arxiv_data(term)
            total_collected += count
            time.sleep(2)
        except Exception as e:
            print(f"  ⚠️  ArXiv error for {term}: {e}")
    
    setup.generate_statistics()
    print(f"✅ Collected {total_collected} real results!")
    return total_collected > 0

def init_real_search_system():
    """Initialize the real search system"""
    try:
        # Create missing __init__.py files if needed
        create_init_files()
        
        from real_search.system import RealSearchSystem

        print("🚀 Initializing Real Search System...")
        system = RealSearchSystem()

        # Test with real queries
        test_queries = ["python", "apple"]
        for query in test_queries:
            results = system.search(query, "en", 3)
            print(f"  ✅ {query}: {len(results)} results")

        return True

    except Exception as e:
        print(f"❌ Error initializing system: {str(e)}")
        print("💡 Make sure you have the real_search package properly set up")
        return False

def create_init_files():
    """Create missing __init__.py files"""
    init_files = [
        "real_search/__init__.py",
        "dataset_manager/__init__.py",
        "dataset_manager/sources/__init__.py"
    ]
    
    init_contents = {
        "real_search/__init__.py": '''"""Real Search Package"""
from .system import RealSearchSystem
from .datasets import DatasetManager
from .clustering import ClusteringEngine
from .feedback import FeedbackProcessor
from .json_utils import NumpyEncoder, clean_for_json

__all__ = ["RealSearchSystem", "DatasetManager", "ClusteringEngine", "FeedbackProcessor", "NumpyEncoder", "clean_for_json"]
''',
        "dataset_manager/__init__.py": '''"""Dataset Manager Package"""
from .core import DatasetManager, DatasetSource
from .database import DatabaseHandler
__all__ = ["DatasetManager", "DatasetSource", "DatabaseHandler"]
''',
        "dataset_manager/sources/__init__.py": '''"""Dataset Sources"""
__all__ = []
'''
    }
    
    for file_path, content in init_contents.items():
        file_path = Path(file_path)
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            print(f"📝 Created {file_path}")

def create_static_files():
    """Ensure static files are in the right place"""
    static_files = ["app.js", "styles.css"]

    for file_name in static_files:
        source = Path(file_name)
        target = Path("static") / file_name

        if source.exists() and not target.exists():
            import shutil
            shutil.copy2(source, target)
            print(f"📄 Copied {file_name} to static folder")

def main():
    """Main setup function"""
    print("🚀 Dynamic Search Disambiguation System - REAL DATA Setup")
    print("=" * 65)
    print("This will collect REAL data from Wikipedia and ArXiv")
    print("Limited to ~10 results per ambiguous term\n")

    # 1. Install requirements
    print("1️⃣ Installing requirements...")
    install_requirements()

    # 2. Setup directories
    print("\n2️⃣ Setting up directories...")
    setup_directories()

    # 3. Create static files
    print("\n3️⃣ Setting up static files...")
    create_static_files()
    
    # 4. Create init files
    print("\n4️⃣ Creating package files...")
    create_init_files()

    # 5. Collect real data
    print("\n5️⃣ Collecting real data...")
    data_success = collect_real_data()

    # 6. Initialize system
    print("\n6️⃣ Testing system...")
    system_success = init_real_search_system()

    # Summary
    print("\n" + "=" * 65)
    if data_success and system_success:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("")
        print("✅ Real data collected from Wikipedia and ArXiv")
        print("✅ System initialized and tested")
        print("")
        print("🚀 To start the system:")
        print("   python app.py")
        print("")
        print("🌐 Then open: http://localhost:5000")
        print("")
        print("🔍 Try these REAL ambiguous queries:")
        print("   • python (programming vs snake)")
        print("   • apple (company vs fruit)")
        print("   • java (programming vs island)")
        print("   • عين (eye vs spring - Arabic)")
    else:
        print("⚠️  SETUP COMPLETED WITH ISSUES")
        print("")
        if not data_success:
            print("❌ Real data collection failed")
            print("   The system will use sample data instead")
        if not system_success:
            print("❌ System initialization failed")
            print("   Check the error messages above")
        print("")
        print("💡 You can still try running: python app.py")
    
    print("=" * 65)

# Add the RealDataSetup class methods
RealDataSetup.init_database = lambda self: self._init_database()
RealDataSetup.fetch_wikipedia_data = lambda self, term, lang: self._fetch_wikipedia_data(term, lang)
RealDataSetup.fetch_arxiv_data = lambda self, term: self._fetch_arxiv_data(term)
RealDataSetup.generate_statistics = lambda self: self._generate_statistics()

def _init_database(self):
    """Initialize the database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_results (
            id TEXT PRIMARY KEY,
            query TEXT, language TEXT, title TEXT, snippet TEXT,
            url TEXT, domain TEXT, category TEXT, dataset_source TEXT,
            relevance_score REAL, embedding TEXT, metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ambiguous_queries (
            id TEXT PRIMARY KEY,
            query TEXT, language TEXT, ambiguity_level REAL,
            entity_types TEXT, dataset_source TEXT, num_meanings INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def _fetch_wikipedia_data(self, term, language):
    """Fetch Wikipedia data for a term"""
    count = 0
    try:
        wiki_base = f"https://{language}.wikipedia.org/api/rest_v1/"
        
        # Try disambiguation page first
        disambig_url = f"{wiki_base}page/summary/{term}_(disambiguation)"
        response = requests.get(disambig_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            self._store_result({
                "query": term, "language": language,
                "title": data["title"], 
                "snippet": data.get("extract", "")[:500],
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "domain": "wikipedia.org", "category": "disambiguation",
                "dataset_source": "wikipedia_disambiguation", "relevance_score": 0.95
            })
            count += 1
        
        # Try main page
        main_url = f"{wiki_base}page/summary/{term}"
        response = requests.get(main_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            self._store_result({
                "query": term, "language": language,
                "title": data["title"],
                "snippet": data.get("extract", "")[:500], 
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "domain": "wikipedia.org", "category": "general",
                "dataset_source": "wikipedia_main", "relevance_score": 0.90
            })
            count += 1
            
        # Try search
        search_url = f"{wiki_base}page/search"
        response = requests.get(search_url, params={"q": term, "limit": 3}, timeout=10)
        
        if response.status_code == 200:
            search_data = response.json()
            for item in search_data.get("pages", [])[:2]:  # Limit to 2
                summary_url = f"{wiki_base}page/summary/{item['key']}"
                summary_response = requests.get(summary_url, timeout=10)
                if summary_response.status_code == 200:
                    data = summary_response.json()
                    self._store_result({
                        "query": term, "language": language,
                        "title": data["title"],
                        "snippet": data.get("extract", "")[:500],
                        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        "domain": "wikipedia.org", "category": "general",
                        "dataset_source": "wikipedia_search", "relevance_score": 0.80
                    })
                    count += 1
                    time.sleep(0.5)
                    
    except Exception as e:
        print(f"    Error fetching {term}: {e}")
    
    return count

def _fetch_arxiv_data(self, term):
    """Fetch ArXiv data for a term"""
    count = 0
    try:
        import urllib.parse
        encoded_query = urllib.parse.quote(term)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&max_results=2"
        
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry")[:2]:
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                link_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                
                if title_elem is not None and summary_elem is not None:
                    self._store_result({
                        "query": term, "language": "en",
                        "title": title_elem.text.strip(),
                        "snippet": summary_elem.text.strip()[:500],
                        "url": link_elem.text if link_elem is not None else "",
                        "domain": "arxiv.org", "category": "academic_paper",
                        "dataset_source": "arxiv_api", "relevance_score": 0.85
                    })
                    count += 1
                    
    except Exception as e:
        print(f"    ArXiv error: {e}")
    
    return count

def _store_result(self, result):
    """Store a result in the database"""
    try:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        result_id = hashlib.md5(f"{result['query']}_{result['title']}".encode()).hexdigest()[:16]
        
        # Generate embedding if model available
        embedding = []
        if self.sentence_model:
            text = f"{result['title']} {result['snippet']}"
            embedding = self.sentence_model.encode(text).tolist()
        
        cursor.execute("""
            INSERT OR REPLACE INTO search_results 
            (id, query, language, title, snippet, url, domain, category,
             dataset_source, relevance_score, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result_id, result["query"], result["language"], result["title"],
            result["snippet"], result["url"], result["domain"], result["category"],
            result["dataset_source"], result["relevance_score"],
            json.dumps(embedding), json.dumps({"real_data": True})
        ))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"    Store error: {e}")
        return False

def _generate_statistics(self):
    """Generate and display statistics"""
    try:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM search_results")
        total_results = cursor.fetchone()[0]
        
        cursor.execute("SELECT dataset_source, COUNT(*) FROM search_results GROUP BY dataset_source")
        source_counts = dict(cursor.fetchall())
        
        cursor.execute("SELECT language, COUNT(*) FROM search_results GROUP BY language")
        language_counts = dict(cursor.fetchall())
        
        conn.close()
        
        print(f"\n📊 REAL DATA STATISTICS:")
        print(f"   Total Results: {total_results}")
        print(f"   Languages: {language_counts}")
        print(f"   Sources: {source_counts}")
        
    except Exception as e:
        print(f"   Stats error: {e}")

# Bind methods to class
RealDataSetup._init_database = _init_database
RealDataSetup._fetch_wikipedia_data = _fetch_wikipedia_data  
RealDataSetup._fetch_arxiv_data = _fetch_arxiv_data
RealDataSetup._store_result = _store_result
RealDataSetup._generate_statistics = _generate_statistics

if __name__ == "__main__":
    main()