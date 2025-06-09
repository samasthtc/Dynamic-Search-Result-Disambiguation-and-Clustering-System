"""
Real Search Disambiguation System - Complete Application
Uses real datasets with Wikipedia, ArXiv, and live API data
Version: 2.1.0
"""

import sys
import os
import logging
import json
import threading
import time
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for system state
search_system = None
system_status = "initializing"
REAL_SEARCH_AVAILABLE = False

# Try to import the real search system
try:
    from real_search.system import RealSearchSystem
    from real_search.json_utils import NumpyEncoder

    REAL_SEARCH_AVAILABLE = True
    logger.info("✅ Real search system imports successful")
except ImportError as e:
    logger.error(f"❌ Failed to import real search system: {e}")
    logger.info("💡 Run: python setup_system.py to set up the system")
    REAL_SEARCH_AVAILABLE = False
    RealSearchSystem = None
    NumpyEncoder = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set JSON encoder if available
if NumpyEncoder:
    app.json_encoder = NumpyEncoder


def initialize_search_system():
    """Initialize the search system with error handling"""
    global search_system, system_status

    if not REAL_SEARCH_AVAILABLE:
        system_status = "unavailable - missing real_search package"
        logger.error("❌ Real search package not available")
        return False

    try:
        logger.info("🚀 Initializing Real Search System...")
        search_system = RealSearchSystem()

        # Quick test to verify system works
        logger.info("🧪 Testing system with sample query...")
        test_results = search_system.search("test", "en", 1)

        # Get system statistics
        stats = search_system.get_dataset_info()
        logger.info(
            f"📊 System loaded with {stats.get('statistics', {}).get('total_results', 0)} results"
        )

        system_status = "ready"
        logger.info("✅ Real Search System initialized successfully!")
        return True

    except Exception as e:
        error_msg = str(e)
        system_status = f"initialization_error - {error_msg}"
        logger.error(f"❌ Failed to initialize search system: {error_msg}")
        return False


def create_fallback_html():
    """Create fallback HTML when index.html is not available"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 Dynamic Search Disambiguation System</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1000px; margin: 0 auto; 
            background: rgba(255, 255, 255, 0.95); 
            padding: 30px; border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #2c3e50; margin-bottom: 10px; }
        .status { 
            padding: 15px; border-radius: 8px; margin: 20px 0; 
            border-left: 4px solid;
        }
        .ready { 
            background: #d4edda; color: #155724; 
            border-left-color: #28a745; 
        }
        .error { 
            background: #f8d7da; color: #721c24; 
            border-left-color: #dc3545; 
        }
        .warning { 
            background: #fff3cd; color: #856404; 
            border-left-color: #ffc107; 
        }
        .search-section {
            background: #f8f9fa; padding: 20px; 
            border-radius: 10px; margin: 20px 0;
        }
        .search-form {
            display: flex; gap: 10px; margin-bottom: 20px;
            flex-wrap: wrap; align-items: center;
        }
        .search-input {
            flex: 1; min-width: 200px; padding: 12px; 
            border: 2px solid #e0e6ff; border-radius: 8px;
            font-size: 16px;
        }
        .search-input:focus {
            outline: none; border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        .btn { 
            background: #6366f1; color: white; 
            padding: 12px 24px; border: none; 
            border-radius: 8px; cursor: pointer; 
            font-size: 16px; font-weight: 600;
            text-decoration: none; display: inline-block;
            transition: all 0.2s ease;
        }
        .btn:hover { 
            background: #4f46e5; 
            transform: translateY(-1px);
        }
        .btn-secondary {
            background: #64748b;
        }
        .btn-secondary:hover {
            background: #475569;
        }
        .samples { 
            background: #e0f2fe; padding: 20px; 
            border-radius: 8px; margin: 20px 0; 
        }
        .samples h3 { margin-top: 0; color: #01579b; }
        .sample-queries {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px; margin-top: 15px;
        }
        .sample-query {
            background: white; padding: 10px; border-radius: 6px;
            border: 1px solid #b3e5fc; cursor: pointer;
            transition: all 0.2s ease;
        }
        .sample-query:hover {
            background: #f0f9ff; border-color: #0288d1;
            transform: translateY(-1px);
        }
        .results-area {
            background: #f8f9fa; padding: 20px;
            border-radius: 10px; margin: 20px 0;
            min-height: 200px; display: none;
        }
        .loading {
            text-align: center; padding: 40px;
            color: #6b7280;
        }
        .spinner {
            border: 3px solid #f3f4f6;
            border-top: 3px solid #6366f1;
            border-radius: 50%; width: 30px; height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .footer {
            text-align: center; margin-top: 30px;
            padding-top: 20px; border-top: 1px solid #e5e7eb;
            color: #6b7280; font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Dynamic Search Result Disambiguation</h1>
            <p>AI-Powered Search Result Clustering with Reinforcement Learning</p>
        </div>
        
        <div class="status {{ 'ready' if status == 'ready' else 'error' if 'error' in status else 'warning' }}">
            <strong>System Status:</strong> {{ status }}
            {% if status == 'ready' %}
                <br><small>✅ Real data loaded from Wikipedia and ArXiv</small>
            {% elif 'error' in status %}
                <br><small>❌ System needs setup. Run: python setup_system.py</small>
            {% else %}
                <br><small>⚠️ System initializing, please wait...</small>
            {% endif %}
        </div>
        
        {% if status == 'ready' %}
        <div class="search-section">
            <h3>🔍 Search & Cluster Ambiguous Queries</h3>
            <div class="search-form">
                <input type="text" id="searchInput" class="search-input" 
                       placeholder="Enter ambiguous query (e.g., 'python', 'apple')..."
                       value="python">
                <select id="languageSelect">
                    <option value="en">🇺🇸 English</option>
                    <option value="ar">🇸🇦 العربية</option>
                </select>
                <button onclick="performSearch()" class="btn">🔍 Search</button>
                <button onclick="performClustering()" class="btn btn-secondary">🎯 Cluster</button>
            </div>
            
            <div id="resultsArea" class="results-area">
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Search for a query to see results...</p>
                </div>
            </div>
        </div>
        {% endif %}
        
        <div class="samples">
            <h3>🔍 Sample Ambiguous Queries</h3>
            <p>Try these real-world ambiguous terms that have multiple meanings:</p>
            <div class="sample-queries">
                <div class="sample-query" onclick="tryQuery('python')">
                    <strong>python</strong><br>
                    <small>Programming language vs snake</small>
                </div>
                <div class="sample-query" onclick="tryQuery('apple')">
                    <strong>apple</strong><br>
                    <small>Technology company vs fruit</small>
                </div>
                <div class="sample-query" onclick="tryQuery('java')">
                    <strong>java</strong><br>
                    <small>Programming language vs island</small>
                </div>
                <div class="sample-query" onclick="tryQuery('mercury')">
                    <strong>mercury</strong><br>
                    <small>Planet vs chemical element</small>
                </div>
                <div class="sample-query" onclick="tryQuery('mars')">
                    <strong>mars</strong><br>
                    <small>Planet vs company vs mythology</small>
                </div>
                <div class="sample-query" onclick="tryQuery('عين', 'ar')">
                    <strong>عين</strong><br>
                    <small>Eye vs spring vs spy (Arabic)</small>
                </div>
            </div>
        </div>
        
        <div style="display: flex; gap: 10px; justify-content: center; margin: 30px 0;">
            <a href="/api/health" class="btn btn-secondary">📊 System Health</a>
            <a href="/api/dataset-info" class="btn btn-secondary">📁 Dataset Info</a>
            <a href="/api/metrics" class="btn btn-secondary">📈 Metrics</a>
            {% if status != 'ready' %}
            <a href="javascript:location.reload()" class="btn">🔄 Refresh</a>
            {% endif %}
        </div>
        
        <div class="footer">
            <p>Real Dataset Search Disambiguation System v2.1.0</p>
            <p>Using real data from Wikipedia, ArXiv, and live APIs</p>
        </div>
    </div>

    <script>
        function tryQuery(query, language = 'en') {
            document.getElementById('searchInput').value = query;
            document.getElementById('languageSelect').value = language;
            performSearch();
        }
        
        function performSearch() {
            const query = document.getElementById('searchInput').value;
            const language = document.getElementById('languageSelect').value;
            const resultsArea = document.getElementById('resultsArea');
            
            if (!query.trim()) {
                alert('Please enter a search query');
                return;
            }
            
            resultsArea.style.display = 'block';
            resultsArea.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Searching for "${query}"...</p>
                </div>
            `;
            
            fetch('/api/search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query, language, num_results: 10})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultsArea.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                resultsArea.innerHTML = `<div class="error">Network error: ${error.message}</div>`;
            });
        }
        
        function performClustering() {
            const resultsArea = document.getElementById('resultsArea');
            
            resultsArea.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Clustering results...</p>
                </div>
            `;
            
            fetch('/api/cluster', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({algorithm: 'kmeans', num_clusters: 3})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultsArea.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                } else {
                    displayClusters(data);
                }
            })
            .catch(error => {
                resultsArea.innerHTML = `<div class="error">Network error: ${error.message}</div>`;
            });
        }
        
        function displayResults(data) {
            const resultsArea = document.getElementById('resultsArea');
            const results = data.results || [];
            
            let html = `
                <h4>🔍 Search Results for "${data.query}" (${results.length} results)</h4>
                <p><small>Sources: ${(data.data_sources || []).join(', ')}</small></p>
            `;
            
            if (results.length === 0) {
                html += '<p>No results found. Try a different query.</p>';
            } else {
                results.forEach((result, idx) => {
                    html += `
                        <div style="background: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #6366f1;">
                            <h5 style="margin: 0 0 8px 0; color: #1f2937;">
                                <a href="${result.url}" target="_blank" style="color: #1f2937; text-decoration: none;">
                                    ${result.title}
                                </a>
                            </h5>
                            <p style="margin: 0 0 8px 0; color: #4b5563; line-height: 1.4;">
                                ${result.snippet}
                            </p>
                            <div style="font-size: 12px; color: #6b7280;">
                                <span>📊 ${result.category}</span> • 
                                <span>🌐 ${result.domain}</span> • 
                                <span>📈 ${(result.relevance_score * 100).toFixed(0)}% relevant</span>
                            </div>
                        </div>
                    `;
                });
                
                html += `
                    <div style="margin-top: 20px; text-align: center;">
                        <button onclick="performClustering()" class="btn">🎯 Cluster These Results</button>
                    </div>
                `;
            }
            
            resultsArea.innerHTML = html;
        }
        
        function displayClusters(data) {
            const resultsArea = document.getElementById('resultsArea');
            const clusters = data.clusters || [];
            
            let html = `
                <h4>🎯 Clustered Results (${clusters.length} clusters)</h4>
                <p><small>Algorithm: ${data.algorithm}</small></p>
            `;
            
            clusters.forEach((cluster, idx) => {
                const clusterColors = ['#e0f2fe', '#f3e5f5', '#e8f5e8', '#fff3e0', '#fce4ec'];
                const color = clusterColors[idx % clusterColors.length];
                
                html += `
                    <div style="background: ${color}; padding: 15px; margin: 15px 0; border-radius: 10px; border: 1px solid #ddd;">
                        <h5 style="margin: 0 0 10px 0; color: #1f2937;">
                            🏷️ ${cluster.label} (${cluster.size} results)
                        </h5>
                `;
                
                cluster.results.forEach(result => {
                    html += `
                        <div style="background: white; padding: 10px; margin: 8px 0; border-radius: 6px;">
                            <strong>${result.title}</strong><br>
                            <small style="color: #4b5563;">${result.snippet.substring(0, 150)}...</small><br>
                            <small style="color: #6b7280;">🌐 ${result.domain} • 📊 ${result.category}</small>
                        </div>
                    `;
                });
                
                html += '</div>';
            });
            
            html += `
                <div style="margin-top: 20px; text-align: center;">
                    <button onclick="performSearch()" class="btn btn-secondary">🔍 New Search</button>
                </div>
            `;
            
            resultsArea.innerHTML = html;
        }
        
        // Auto-refresh if system not ready
        {% if status != 'ready' %}
        setTimeout(() => location.reload(), 5000);
        {% endif %}
    </script>
</body>
</html>
    """


@app.route("/")
def index():
    """Serve the main HTML interface"""
    try:
        # Try to serve static index.html first
        """Serve the main HTML interface"""
        return app.send_static_file("index.html")
    except Exception:
        # Fallback to dynamic HTML
        return render_template_string(create_fallback_html(), status=system_status)


@app.route("/api/search", methods=["POST"])
def api_search():
    """Perform search using real datasets"""
    if not search_system:
        return (
            jsonify(
                {
                    "error": "Search system not available",
                    "status": system_status,
                    "suggestion": "Run: python setup_system.py to initialize the system",
                }
            ),
            500,
        )

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), 400

        query = data.get("query", "").strip()
        language = data.get("language", "en")
        num_results = data.get("num_results", 20)

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        logger.info(
            f"🔍 Search request: '{query}' ({language}) - {num_results} results"
        )

        # Perform search
        results = search_system.search(query, language, num_results)

        response_data = {
            "status": "success",
            "results": results,
            "query": query,
            "language": language,
            "total_results": len(results),
            "data_sources": search_system.get_last_search_sources(),
            "is_real_data": True,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"✅ Search completed: {len(results)} results found")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ Search error: {str(e)}")
        return (
            jsonify(
                {
                    "error": str(e),
                    "query": data.get("query", "") if data else "",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


@app.route("/api/cluster", methods=["POST"])
def api_cluster():
    """Cluster current search results"""
    if not search_system:
        return (
            jsonify({"error": "Search system not available", "status": system_status}),
            500,
        )

    try:
        data = request.get_json() or {}
        algorithm = data.get("algorithm", "kmeans")
        num_clusters = data.get("num_clusters", 4)
        min_cluster_size = data.get("min_cluster_size", 2)

        logger.info(f"🎯 Clustering request: {algorithm} with {num_clusters} clusters")

        clusters = search_system.cluster(algorithm, num_clusters, min_cluster_size)

        response_data = {
            "status": "success",
            "clusters": clusters,
            "algorithm": algorithm,
            "total_clusters": len(clusters),
            "is_real_data": True,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"✅ Clustering completed: {len(clusters)} clusters created")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ Clustering error: {str(e)}")
        return jsonify({"error": str(e), "timestamp": datetime.now().isoformat()}), 500


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Process user feedback"""
    if not search_system:
        return jsonify({"error": "Search system not available"}), 500

    try:
        feedback_data = request.get_json()
        if not feedback_data:
            return jsonify({"error": "Feedback data is required"}), 400

        logger.info(f"💬 Feedback received: {feedback_data.get('feedback', 'unknown')}")
        result = search_system.process_feedback(feedback_data)

        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Feedback error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Get system performance metrics"""
    if not search_system:
        return jsonify(
            {
                "error": "Search system not available",
                "total_queries": 0,
                "system_status": system_status,
            }
        )

    try:
        metrics = search_system.get_metrics()
        metrics["system_status"] = system_status
        metrics["timestamp"] = datetime.now().isoformat()
        return jsonify(metrics)

    except Exception as e:
        logger.error(f"❌ Metrics error: {str(e)}")
        return jsonify({"error": str(e), "system_status": system_status}), 500


@app.route("/api/dataset-info", methods=["GET"])
def api_dataset_info():
    """Get information about available datasets"""
    if not search_system:
        return jsonify(
            {
                "error": "Search system not available",
                "status": system_status,
                "available_sources": [],
                "suggestion": "Run: python setup_system.py",
            }
        )

    try:
        dataset_info = search_system.get_dataset_info()
        dataset_info["system_status"] = system_status
        dataset_info["timestamp"] = datetime.now().isoformat()
        return jsonify(dataset_info)

    except Exception as e:
        logger.error(f"❌ Dataset info error: {str(e)}")
        return jsonify({"error": str(e), "system_status": system_status}), 500


@app.route("/api/ambiguous-queries", methods=["GET"])
def api_ambiguous_queries():
    """Get real ambiguous queries from datasets"""
    if not search_system:
        return jsonify(
            {
                "error": "Search system not available",
                "queries": [],
                "status": system_status,
            }
        )

    try:
        language = request.args.get("language", "en")
        limit = int(request.args.get("limit", 20))

        queries = search_system.get_ambiguous_queries(language, limit)

        return jsonify(
            {
                "status": "success",
                "queries": queries,
                "language": language,
                "total_queries": len(queries),
                "is_real_data": True,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"❌ Ambiguous queries error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Comprehensive health check endpoint"""
    try:
        health_info = {
            "status": "healthy" if search_system else "degraded",
            "system_status": system_status,
            "timestamp": datetime.now().isoformat(),
            "system_type": "real_datasets",
            "version": "2.1.0",
            "real_search_available": REAL_SEARCH_AVAILABLE,
            "components": {
                "search_system": search_system is not None,
                "real_data": REAL_SEARCH_AVAILABLE,
            },
        }

        if search_system:
            try:
                # Get additional system info
                dataset_info = search_system.get_dataset_info()
                health_info.update(
                    {
                        "datasets_loaded": search_system.get_loaded_datasets(),
                        "last_search_sources": search_system.get_last_search_sources(),
                        "total_results": dataset_info.get("statistics", {}).get(
                            "total_results", 0
                        ),
                        "data_sources": dataset_info.get("statistics", {}).get(
                            "results_by_source", {}
                        ),
                        "languages": dataset_info.get("statistics", {}).get(
                            "results_by_language", {}
                        ),
                    }
                )
            except Exception as e:
                health_info["warning"] = f"Could not get detailed stats: {str(e)}"

        status_code = 200 if search_system else 503
        return jsonify(health_info), status_code

    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "system_status": system_status,
                }
            ),
            500,
        )


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files"""
    try:
        return send_from_directory("static", filename)
    except Exception:
        return "Static file not found", 404


def periodic_save():
    """Background thread for periodic system state saving"""
    while True:
        time.sleep(300)  # Save every 5 minutes
        try:
            if search_system:
                search_system.save_state()
                logger.debug("💾 Periodic save completed")
        except Exception as e:
            logger.error(f"❌ Periodic save error: {str(e)}")


def main():
    """Main application entry point"""
    global search_system, system_status

    # Initialize system at startup
    logger.info("🚀 Starting Real Data Search Disambiguation System...")
    initialize_search_system()

    # Start background save thread
    if search_system:
        save_thread = threading.Thread(target=periodic_save, daemon=True)
        save_thread.start()
        logger.info("📝 Background save thread started")

    # Print startup banner
    print("\n" + "=" * 70)
    print("🔍 DYNAMIC SEARCH RESULT DISAMBIGUATION SYSTEM")
    print("=" * 70)
    print(f"Status: {'✅ READY' if search_system else '❌ NOT READY'}")
    print(f"System: {system_status}")
    print("")

    if search_system:
        try:
            dataset_info = search_system.get_dataset_info()
            stats = dataset_info.get("statistics", {})
            print("📊 Real Data Loaded:")
            print(f"   • Total Results: {stats.get('total_results', 0)}")
            print(
                f"   • Languages: {list(stats.get('results_by_language', {}).keys())}"
            )
            print(f"   • Sources: {list(stats.get('results_by_source', {}).keys())}")
            print("")
            print("🔍 Try these ambiguous queries:")
            print("   • python (programming vs snake)")
            print("   • apple (company vs fruit)")
            print("   • java (programming vs island)")
            print("   • mercury (planet vs element)")
            print("   • عين (eye vs spring - Arabic)")
        except Exception:
            print("⚠️  System loaded but stats unavailable")
    else:
        print("❌ System not ready. To fix:")
        print("   1. Run: python setup_system.py")
        print("   2. Check error messages above")
        print("   3. Ensure all dependencies are installed")

    print("")
    print("🌐 Server will start at: http://localhost:5000")
    print("📊 Health check: http://localhost:5000/api/health")
    print("📁 Dataset info: http://localhost:5000/api/dataset-info")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()

        # Run the Flask app
        app.run(
            debug=True,
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False,  # Prevent double initialization
        )

    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        if search_system:
            try:
                search_system.save_state()
                print("💾 System state saved")
            except Exception:
                pass
    except Exception as e:
        logger.error(f"❌ Application error: {str(e)}")
        print(f"\n❌ Application failed to start: {str(e)}")
        print("\n💡 Troubleshooting steps:")
        print("1. Run: python setup_system.py")
        print("2. Check that all dependencies are installed")
        print("3. Verify the real_search package is available")
        sys.exit(1)
