/**
 * Dynamic Search Result Disambiguation System
 * Frontend JavaScript Application - FIXED VERSION (Event Listeners)
 */

class SearchDisambiguationAPI {
  constructor() {
    this.baseURL = 'http://localhost:5000/api';
    this.currentResults = [];
    this.currentClusters = [];
    this.currentQuery = '';
    this.currentLanguage = 'en';

    this.initializeEventListeners();
    this.loadInitialMetrics();
    this.setupPeriodicUpdates();
  }

  /**
   * Initialize all event listeners for the interface
   */
  initializeEventListeners() {
    // Search functionality
    document
      .getElementById('searchBtn')
      .addEventListener('click', () => this.performSearch());
    document.getElementById('searchInput').addEventListener('keypress', e => {
      if (e.key === 'Enter') this.performSearch();
    });

    // Language selection
    document.getElementById('languageSelect').addEventListener('change', e => {
      this.currentLanguage = e.target.value;
    });

    // Clustering controls
    document.getElementById('clusterSlider').addEventListener('input', e => {
      document.getElementById('clusterValue').textContent = e.target.value;
    });

    document
      .getElementById('reclusterBtn')
      .addEventListener('click', () => this.reclusterResults());
    document
      .getElementById('ensembleBtn')
      .addEventListener('click', () => this.performEnsembleClustering());

    // Add sample query buttons
    this.addSampleQueryButtons();
  }

  /**
   * Add sample query buttons for quick testing
   */
  addSampleQueryButtons() {
    const sampleQueries = [
      { query: 'Jackson', desc: 'Person/Place ambiguity' },
      { query: 'Apple', desc: 'Company/Fruit ambiguity' },
      { query: 'Python', desc: 'Programming/Animal ambiguity' },
      { query: 'Mercury', desc: 'Planet/Element ambiguity' },
    ];

    const searchInput = document.getElementById('searchInput');

    // Create sample buttons container
    const samplesDiv = document.createElement('div');
    samplesDiv.style.cssText =
      'margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap;';

    sampleQueries.forEach(sample => {
      const btn = document.createElement('button');
      btn.textContent = `Try "${sample.query}"`;
      btn.title = sample.desc;
      btn.style.cssText = `
                padding: 8px 16px; 
                background: #f1f5f9; 
                border: 1px solid #e2e8f0; 
                border-radius: 20px; 
                cursor: pointer;
                font-size: 13px;
                transition: all 0.2s ease;
            `;

      btn.addEventListener('click', () => {
        searchInput.value = sample.query;
        this.performSearch();
      });

      btn.addEventListener('mouseenter', () => {
        btn.style.background = '#6366f1';
        btn.style.color = 'white';
      });

      btn.addEventListener('mouseleave', () => {
        btn.style.background = '#f1f5f9';
        btn.style.color = 'inherit';
      });

      samplesDiv.appendChild(btn);
    });

    searchInput.parentNode.insertBefore(samplesDiv, searchInput.nextSibling);
  }

  /**
   * Perform search with the current query
   */
  async performSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) {
      this.showMessage('Please enter a search query', 'error');
      return;
    }

    this.currentQuery = query;
    this.showLoading();

    try {
      const response = await fetch(`${this.baseURL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          language: this.currentLanguage,
          num_results: 20,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.currentResults = data.results;
      this.displayOriginalResults();

      // Automatically perform clustering
      await this.performClustering();

      this.showMessage(
        `Found ${data.total_results} results for "${query}"`,
        'success'
      );
    } catch (error) {
      console.error('Search error:', error);
      this.showMessage(
        'Search failed. Please check if the backend server is running.',
        'error'
      );
    }
  }

  /**
   * Perform clustering on current results
   */
  async performClustering() {
    if (this.currentResults.length === 0) return;

    const algorithm = document.getElementById('algorithmSelect').value;
    const numClusters = parseInt(
      document.getElementById('clusterSlider').value
    );

    try {
      const response = await fetch(`${this.baseURL}/cluster`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          algorithm: algorithm,
          num_clusters: numClusters,
          min_cluster_size: 2,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.currentClusters = data.clusters;
      this.displayClusteredResults();

      // Update metrics
      await this.updateMetrics();
    } catch (error) {
      console.error('Clustering error:', error);
      this.showMessage('Clustering failed. Please try again.', 'error');
    }
  }

  /**
   * Perform ensemble clustering using multiple algorithms
   */
  async performEnsembleClustering() {
    if (this.currentResults.length === 0) {
      this.showMessage('Please perform a search first', 'error');
      return;
    }

    document.getElementById('clusteredResults').innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Running ensemble clustering (multiple algorithms)...</p>
            </div>
        `;

    try {
      const response = await fetch(`${this.baseURL}/cluster`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          algorithm: 'ensemble',
          num_clusters: parseInt(
            document.getElementById('clusterSlider').value
          ),
          ensemble_algorithms: ['kmeans', 'hdbscan', 'gaussian_mixture'],
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.currentClusters = data.clusters;
      this.displayClusteredResults();

      this.showMessage(
        'Ensemble clustering completed using multiple algorithms',
        'success'
      );
    } catch (error) {
      console.error('Ensemble clustering error:', error);
      this.showMessage(
        'Ensemble clustering failed. Please try again.',
        'error'
      );
    }
  }

  /**
   * Re-cluster results with updated parameters
   */
  async reclusterResults() {
    if (this.currentResults.length === 0) {
      this.showMessage('Please perform a search first', 'error');
      return;
    }

    document.getElementById('clusteredResults').innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Re-clustering with updated parameters...</p>
            </div>
        `;

    await this.performClustering();
  }

  /**
   * Show loading state in both result panels
   */
  showLoading() {
    document.getElementById('originalResults').innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Searching and analyzing results...</p>
            </div>
        `;

    document.getElementById('clusteredResults').innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Preparing AI clustering...</p>
            </div>
        `;
  }

  /**
   * Display original search results
   */
  displayOriginalResults() {
    const container = document.getElementById('originalResults');
    container.innerHTML = '';

    this.currentResults.forEach((result, index) => {
      const resultDiv = document.createElement('div');
      resultDiv.className = 'result-item';
      resultDiv.innerHTML = `
                <div class="result-title">${this.escapeHtml(result.title)}</div>
                <div class="result-snippet">${this.escapeHtml(
                  result.snippet
                )}</div>
                <a href="${result.url}" class="result-url" target="_blank">${
        result.url
      }</a>
                <div class="feedback-section">
                    <strong>Rate this result:</strong>
                    <div class="feedback-buttons">
                        <button class="feedback-btn" data-index="${index}" data-feedback="relevant" data-context="result">👍 Relevant</button>
                        <button class="feedback-btn" data-index="${index}" data-feedback="irrelevant" data-context="result">👎 Irrelevant</button>
                        <button class="feedback-btn" data-index="${index}" data-feedback="wrong_cluster" data-context="result">🔄 Wrong Cluster</button>
                    </div>
                </div>
            `;

      // Add event listeners to feedback buttons
      const feedbackButtons = resultDiv.querySelectorAll('.feedback-btn');
      feedbackButtons.forEach(button => {
        button.addEventListener('click', e => {
          const index = parseInt(e.target.dataset.index);
          const feedback = e.target.dataset.feedback;
          const context = e.target.dataset.context;
          this.provideFeedback(e, index, feedback, context);
        });
      });

      container.appendChild(resultDiv);
    });
  }

  /**
   * Display clustered results
   */
  displayClusteredResults() {
    const container = document.getElementById('clusteredResults');
    container.innerHTML = '';

    this.currentClusters.forEach((cluster, clusterIndex) => {
      const clusterDiv = document.createElement('div');
      clusterDiv.className = 'cluster';
      clusterDiv.innerHTML = `
                <div class="cluster-header">
                    <span>${this.escapeHtml(cluster.label)}</span>
                    <span class="cluster-stats">${cluster.size} results</span>
                </div>
                <div class="cluster-content">
                    ${cluster.results
                      .map(
                        result => `
                        <div class="result-item">
                            <div class="result-title">${this.escapeHtml(
                              result.title
                            )}</div>
                            <div class="result-snippet">${this.escapeHtml(
                              result.snippet
                            )}</div>
                            <a href="${
                              result.url
                            }" class="result-url" target="_blank">${
                          result.url
                        }</a>
                        </div>
                    `
                      )
                      .join('')}
                    <div class="feedback-section">
                        <strong>Rate this cluster:</strong>
                        <div class="feedback-buttons">
                            <button class="feedback-btn" data-cluster-index="${clusterIndex}" data-feedback="excellent">⭐ Excellent</button>
                            <button class="feedback-btn" data-cluster-index="${clusterIndex}" data-feedback="good">👍 Good</button>
                            <button class="feedback-btn" data-cluster-index="${clusterIndex}" data-feedback="poor">👎 Poor</button>
                            <button class="feedback-btn" data-cluster-index="${clusterIndex}" data-feedback="should_split">✂️ Split</button>
                            <button class="feedback-btn" data-cluster-index="${clusterIndex}" data-feedback="should_merge">🔗 Merge</button>
                        </div>
                    </div>
                </div>
            `;

      // Add event listeners to cluster feedback buttons
      const clusterFeedbackButtons = clusterDiv.querySelectorAll(
        '.cluster-feedback-btn'
      );
      clusterFeedbackButtons.forEach(button => {
        button.addEventListener('click', e => {
          const clusterIndex = parseInt(e.target.dataset.clusterIndex);
          const feedback = e.target.dataset.feedback;
          this.provideClusterFeedback(e, clusterIndex, feedback);
        });
      });

      container.appendChild(clusterDiv);
    });
  }

  /**
   * Provide feedback for individual search results
   * FIXED: Now properly handles event object
   */
  async provideFeedback(event, resultIndex, feedback, context = 'result') {
    const feedbackData = {
      result_index: resultIndex,
      feedback: feedback,
      context: context,
      query: this.currentQuery,
      language: this.currentLanguage,
    };

    try {
      const response = await fetch(`${this.baseURL}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      // Visual feedback - now properly using the event parameter
      if (event && event.target) {
        event.target.classList.add('active');
        setTimeout(() => event.target.classList.remove('active'), 1000);
      }

      // Update RL status
      this.updateRLStatus(result);

      // Update metrics
      await this.updateMetrics();
    } catch (error) {
      console.error('Feedback error:', error);
      this.showMessage('Failed to submit feedback', 'error');
    }
  }

  /**
   * Provide feedback for cluster quality
   * FIXED: Now properly handles event object
   */
  async provideClusterFeedback(event, clusterIndex, feedback) {
    const feedbackData = {
      cluster_index: clusterIndex,
      feedback: feedback,
      context: 'cluster',
      query: this.currentQuery,
      cluster_size: this.currentClusters[clusterIndex]?.size || 0,
    };

    try {
      const response = await fetch(`${this.baseURL}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      // Visual feedback - now properly using the event parameter
      if (event && event.target) {
        event.target.classList.add('active');
        setTimeout(() => event.target.classList.remove('active'), 1000);
      }

      // Update RL status
      this.updateRLStatus(result);

      // Update metrics
      await this.updateMetrics();
    } catch (error) {
      console.error('Cluster feedback error:', error);
      this.showMessage('Failed to submit cluster feedback', 'error');
    }
  }

  /**
   * Update RL agent status display
   */
  updateRLStatus(result) {
    document.getElementById('episodes').textContent =
      result.total_episodes || 0;
    document.getElementById('totalReward').textContent = (
      result.reward || 0
    ).toFixed(1);
    document.getElementById('explorationRate').textContent =
      Math.round((result.exploration_rate || 0) * 100) + '%';

    const statusMessages = [
      'Learning patterns...',
      'Optimizing clusters...',
      'Adapting strategy...',
      'Processing feedback...',
      'Improving results...',
    ];

    const randomStatus =
      statusMessages[Math.floor(Math.random() * statusMessages.length)];
    document.getElementById('rlStatus').textContent = randomStatus;
  }

  /**
   * Update performance metrics display
   */
  async updateMetrics() {
    try {
      const response = await fetch(`${this.baseURL}/metrics`);
      if (!response.ok) return;

      const metrics = await response.json();

      document.getElementById('purityScore').textContent = (
        metrics.cluster_purity || 0
      ).toFixed(2);
      document.getElementById('randIndex').textContent = (
        metrics.adjusted_rand_index || 0
      ).toFixed(2);
      document.getElementById('silhouetteScore').textContent = (
        metrics.silhouette_score || 0
      ).toFixed(2);
      document.getElementById('userSatisfaction').textContent =
        (metrics.user_satisfaction_pct || 0) + '%';
      document.getElementById('totalQueries').textContent =
        metrics.total_queries || 0;
      document.getElementById('feedbackItems').textContent =
        metrics.total_feedback_items || 0;
    } catch (error) {
      console.error('Metrics update error:', error);
    }
  }

  /**
   * Load initial metrics on page load
   */
  async loadInitialMetrics() {
    await this.updateMetrics();
  }

  /**
   * Setup periodic updates for metrics
   */
  setupPeriodicUpdates() {
    // Update metrics every 30 seconds
    setInterval(() => {
      this.updateMetrics();
    }, 30000);
  }

  /**
   * Show user messages (success/error)
   */
  showMessage(message, type = 'info') {
    // Remove existing messages
    const existingMessages = document.querySelectorAll(
      '.error-message, .success-message'
    );
    existingMessages.forEach(msg => msg.remove());

    const messageDiv = document.createElement('div');
    messageDiv.className =
      type === 'error' ? 'error-message' : 'success-message';
    messageDiv.textContent = message;

    const mainContent = document.querySelector('.main-content');
    mainContent.insertBefore(messageDiv, mainContent.firstChild);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      messageDiv.remove();
    }, 5000);
  }

  /**
   * Escape HTML to prevent XSS
   */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
  // Initialize the main API class
  const api = new SearchDisambiguationAPI();

  // Make API instance globally available for onclick handlers
  window.api = api;

  // Set initial sample query
  document.getElementById('searchInput').value = 'Jackson';

  console.log('Search Disambiguation System initialized successfully');
});
