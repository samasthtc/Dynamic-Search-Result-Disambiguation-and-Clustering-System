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
    this.addSampleQueryButtons();
  }

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
      this.updateLanguageInterface();
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
  }

  addSampleQueryButtons() {
    const sampleQueries = [
      { query: 'Jackson', desc: 'Person/Place ambiguity' },
      { query: 'Apple', desc: 'Company/Fruit ambiguity' },
      { query: 'Python', desc: 'Programming/Animal ambiguity' },
      { query: 'Mercury', desc: 'Planet/Element ambiguity' },
      { query: 'عين', desc: 'Arabic: Eye' },
      { query: 'تفاحة', desc: 'Arabic: Apple' },
    ];

    const searchContainer = document.querySelector('.search-container');

    // Add sample buttons below search
    const samplesDiv = document.createElement('div');
    samplesDiv.className = 'sample-queries';

    sampleQueries.forEach(sample => {
      const btn = document.createElement('button');
      btn.textContent = `Try "${sample.query}"`;
      btn.title = sample.desc;
      btn.className = 'sample-btn';

      btn.addEventListener('click', () => {
        document.getElementById('searchInput').value = sample.query;
        this.performSearch();
      });

      samplesDiv.appendChild(btn);
    });

    searchContainer.parentNode.insertBefore(
      samplesDiv,
      searchContainer.nextSibling
    );
  }

  updateLanguageInterface() {
    const container = document.querySelector('.main-content');
    const searchInput = document.getElementById('searchInput');

    if (this.currentLanguage === 'ar') {
      // container.classList.add('arabic-support');
      searchInput.placeholder =
        'أدخل استعلام البحث الغامض (مثل: "جاكسون"، "تفاحة"، "بايثون")...';
    } else {
      container.classList.remove('arabic-support');
      searchInput.placeholder =
        'Enter ambiguous search query (e.g., "Jackson", "Apple", "Python")...';
    }
  }

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
          num_results: 10,
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
                        <button class="feedback-btn" onclick="api.provideFeedback(event, ${index}, 'relevant', 'result')">👍 Relevant</button>
                        <button class="feedback-btn" onclick="api.provideFeedback(event, ${index}, 'irrelevant', 'result')">👎 Irrelevant</button>
                        <button class="feedback-btn" onclick="api.provideFeedback(event, ${index}, 'wrong_cluster', 'result')">🔄 Wrong Cluster</button>
                    </div>
                </div>
            `;
      container.appendChild(resultDiv);
    });
  }

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
                            <button class="feedback-btn" onclick="api.provideClusterFeedback(event, ${clusterIndex}, 'excellent')">⭐ Excellent</button>
                            <button class="feedback-btn" onclick="api.provideClusterFeedback(event, ${clusterIndex}, 'good')">👍 Good</button>
                            <button class="feedback-btn" onclick="api.provideClusterFeedback(event, ${clusterIndex}, 'poor')">👎 Poor</button>
                            <button class="feedback-btn" onclick="api.provideClusterFeedback(event, ${clusterIndex}, 'should_split')">✂️ Split</button>
                            <button class="feedback-btn" onclick="api.provideClusterFeedback(event, ${clusterIndex}, 'should_merge')">🔗 Merge</button>
                        </div>
                    </div>
                </div>
            `;
      container.appendChild(clusterDiv);
    });
  }

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

      // Visual feedback
      event.target.classList.add('active');
      setTimeout(() => event.target.classList.remove('active'), 1000);

      // Update RL status
      this.updateRLStatus(result);

      // Update metrics
      await this.updateMetrics();
    } catch (error) {
      console.error('Feedback error:', error);
      this.showMessage('Failed to submit feedback', 'error');
    }
  }

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

      // Visual feedback
      event.target.classList.add('active');
      setTimeout(() => event.target.classList.remove('active'), 1000);

      // Update RL status
      this.updateRLStatus(result);

      // Update metrics
      await this.updateMetrics();
    } catch (error) {
      console.error('Cluster feedback error:', error);
      this.showMessage('Failed to submit cluster feedback', 'error');
    }
  }

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
        (metrics.user_satisfaction_pct.toFixed(2) || 0) + '%';
      document.getElementById('totalQueries').textContent =
        metrics.total_queries || 0;
      document.getElementById('feedbackItems').textContent =
        metrics.total_feedback_items || 0;
    } catch (error) {
      console.error('Metrics update error:', error);
    }
  }

  async loadInitialMetrics() {
    await this.updateMetrics();
  }

  setupPeriodicUpdates() {
    // Update metrics every 30 seconds
    setInterval(() => {
      this.updateMetrics();
    }, 30000);
  }

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

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize the application
const api = new SearchDisambiguationAPI();

// Make API instance globally available for onclick handlers
window.api = api;

// Set initial sample query
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('searchInput').value = 'Jackson';
});
