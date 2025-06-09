#!/usr/bin/env python3
"""
Complete Real Data Setup Script
This script will:
1. Install required packages
2. Collect real data from Wikipedia and ArXiv
3. Set up the system for real data usage
4. Test the system
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def install_packages():
    """Install required packages"""
    print("📦 Installing required packages...")

    packages = [
        "sentence-transformers",
        "requests",
        "flask",
        "flask-cors",
        "numpy",
        "pandas",
        "scikit-learn",
    ]

    optional_packages = ["hdbscan", "wikipedia-api", "arabic-reshaper", "python-bidi"]

    # Install core packages
    for package in packages:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"  ❌ {package} - failed")

    # Install optional packages
    for package in optional_packages:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  {package} - optional, skipped")


def collect_real_data():
    """Collect real data using the collector script"""
    print("\n🔍 Collecting real data from Wikipedia and ArXiv...")
    print("This may take a few minutes...")

    try:
        # Import and run the data collector
        from setup_real_data import RealDataCollector

        collector = RealDataCollector(max_results_per_term=10)
        total_collected = collector.collect_all_real_data()

        if total_collected > 0:
            print(f"✅ Successfully collected {total_collected} real results!")
            return True
        else:
            print("❌ No data collected")
            return False

    except Exception as e:
        print(f"❌ Error collecting data: {e}")
        return False


def test_system():
    """Test the system with real data"""
    print("\n🧪 Testing the system...")

    try:
        from real_search.system import RealSearchSystem

        # Initialize system
        system = RealSearchSystem()

        # Test searches with known ambiguous terms
        test_queries = [("python", "en"), ("apple", "en"), ("عين", "ar")]

        all_working = True

        for query, language in test_queries:
            try:
                results = system.search(query, language, 5)
                if results:
                    print(f"  ✅ {query} ({language}): {len(results)} results")

                    # Test clustering
                    clusters = system.cluster("kmeans", 2, 1)
                    if clusters:
                        print(f"    🎯 Clustering: {len(clusters)} clusters")

                else:
                    print(f"  ⚠️  {query} ({language}): No results")

            except Exception as e:
                print(f"  ❌ {query} ({language}): Error - {e}")
                all_working = False

        return all_working

    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False


def create_startup_script():
    """Create a simple startup script"""
    startup_content = '''#!/usr/bin/env python3
"""
Quick Start Script for Real Data Search System
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    print("🚀 Starting Real Data Search Disambiguation System...")
    
    # Import and run the app
    from app import app, search_system
    
    if search_system:
        print("✅ System loaded successfully!")
        print("🌐 Opening on: http://localhost:5000")
        print("📊 Check /api/health for system status")
        print("🔍 Try searching for: python, apple, عين")
        print("")
        
        app.run(debug=False, host="0.0.0.0", port=5000)
    else:
        print("❌ System failed to load")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Try running: python run_real_data_setup.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
'''

    with open("start_system.py", "w") as f:
        f.write(startup_content)

    print("📝 Created start_system.py")


def main():
    """Main setup function"""
    print("🚀 Real Data Search Disambiguation System Setup")
    print("=" * 60)
    print("This will set up the system to work with REAL data from:")
    print("📚 Wikipedia disambiguation pages")
    print("🔬 ArXiv academic papers")
    print("🌐 Live API fetching")
    print("")

    # Step 1: Install packages
    install_packages()

    # Step 2: Collect real data
    if collect_real_data():
        print("✅ Real data collection successful!")
    else:
        print("⚠️  Real data collection failed, but system will work with sample data")

    # Step 3: Test system
    if test_system():
        print("✅ System test successful!")
    else:
        print("⚠️  System test had issues, but basic functionality should work")

    # Step 4: Create startup script
    create_startup_script()

    # Final instructions
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("")
    print("🚀 To start the system:")
    print("   python start_system.py")
    print("   OR")
    print("   python app.py")
    print("")
    print("🌐 Then open: http://localhost:5000")
    print("")
    print("🔍 Try these ambiguous queries:")
    print("   - python (programming vs snake)")
    print("   - apple (company vs fruit)")
    print("   - عين (eye vs spring)")
    print("")
    print("📊 Check system status at: http://localhost:5000/api/health")
    print("=" * 60)


if __name__ == "__main__":
    main()
