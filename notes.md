# DSR-RL Presentation Speaker Notes (7 Minutes)

## Slide 1: Title Slide (30 seconds)

- "Good morning, Professor. Today we're presenting DSR-RL: Dynamic Search-Result Disambiguation and Clustering via Reinforcement Learning."
- "This is joint work by our team: Shatha Khdair, Usama Shoora, Mohammad AbuSalah, and myself."
- "We'll show you how we solved the problem of ambiguous search results using AI that learns from user feedback."

## Slide 2: Problem Overview (45 seconds)

- "The problem we're solving is fundamental to information retrieval. When users search for 'Jackson,' they get mixed results about Michael Jackson the singer, Andrew Jackson the president, and Jackson Mississippi the city."
- "Users waste time manually sorting through irrelevant results. This problem is worse in Arabic due to diacritics - the same word can be written multiple ways."
- "For example, محمد can be written with or without diacritics, creating additional ambiguity."

## Slide 3: Why Static Clustering Fails (30 seconds)

- "Traditional clustering uses fixed parameters that can't adapt to different user needs or query contexts."
- "There's no learning mechanism - the system makes the same mistakes repeatedly."
- "Most importantly, existing systems lack proper Arabic language support, missing the linguistic complexities we mentioned."

## Slide 4: Our Solution - DSR-RL (45 seconds)

- "Our solution combines three powerful technologies: Sentence-BERT for semantic understanding, BERTopic for interpretable clustering, and Reinforcement Learning for adaptation."
- "Instead of static rules, our system learns what good clustering looks like from user feedback."
- "When users merge clusters or split them, the system learns and improves future clustering decisions."

## Slide 5: System Workflow (60 seconds)

- "Our pipeline has five stages. First, we preprocess the query and snippets - this is crucial for Arabic text normalization."
- "Second, we generate embeddings using either Sentence-BERT for semantic similarity or TF-IDF for keyword-based similarity."
- "Third, our RL agent selects the best clustering algorithm and parameters based on the current state."
- "Fourth, we present clustered results in our web interface with clear groupings."
- "Finally, we collect user feedback through relevance ratings and merge/split actions, feeding this back to our RL agent."

## Slide 6: Text Preprocessing (45 seconds)

- "Arabic preprocessing is critical for our system's success. We remove diacritics to normalize different spellings of the same word."
- "We unify Hamza and Alif variants - these are different ways of writing the same sound in Arabic."
- "We use Farasa tokenizer for Arabic and spaCy for English. Our light stemming reduces Arabic vocabulary by 23%, improving clustering efficiency."

## Slide 7: Clustering Techniques (45 seconds)

- "We implement five clustering algorithms: K-Means for spherical clusters, DBSCAN for density-based clustering, HDBSCAN for hierarchical density clustering, GMM for probabilistic clustering, and BERTopic for topic-based clustering."
- "BERTopic is our preferred method because it provides interpretable topic labels using c-TF-IDF, making results more understandable to users."
- "Our RL agent dynamically switches between algorithms based on data characteristics like density and dimensionality."

## Slide 8: Default Clustering Parameters (30 seconds)

- "These are our baseline parameters before RL optimization. K-Means uses elbow method for k selection, DBSCAN uses ε=0.7, HDBSCAN requires minimum 5 samples, and BERTopic needs at least 10 documents per topic."
- "The RL agent learns to adjust these parameters based on performance feedback."

## Slide 9: Reinforcement Learning Setup (60 seconds)

- "Our RL formulation uses a 4-dimensional state space: query length indicates complexity, vector density measures semantic similarity, sparsity indicates keyword diversity, and Jensen-Shannon divergence measures result heterogeneity."
- "Actions combine embedding methods with clustering algorithms and their parameters."
- "Rewards combine automatic metrics like silhouette score with user feedback. Positive feedback for good clusters, negative for poor ones, and bonuses for user actions like successful merges."

## Slide 10: Clustering Evaluation Metrics (30 seconds)

- "We evaluate using standard clustering metrics. Silhouette measures cluster cohesion, Purity measures homogeneity, ARI measures agreement with ground truth, and Topic Coherence measures interpretability."

## Slide 11: Metric Comparison (45 seconds)

- "Our results show significant improvements over static BERTopic. Silhouette improved from 0.38 to 0.46, Purity increased 13% from 0.71 to 0.80, ARI increased 23% from 0.44 to 0.54, and coherence improved from 0.29 to 0.37."
- "These improvements demonstrate that RL adaptation significantly enhances clustering quality."

## Slide 12: Experimental Setup (30 seconds)

- "We tested on three datasets: TREC Web Diversity for English ambiguous queries, MIRACL-Arabic for Arabic IR evaluation, and Wikipedia disambiguation pages."
- "Using commodity hardware - Intel i7 with 16GB RAM, no GPU - we achieved sub-second latency: 480ms for embedding, 220ms for clustering, total 920ms."

## Slide 13: Raw List of Results (15 seconds)

- "This shows the problem - mixed results about different Jacksons with no organization, forcing users to manually scan through everything."

## Slide 14: Clustered Search Results (15 seconds)

- "After our processing, results are clearly organized by topic - here we see 'Artists & Creators' cluster with Jackson Pollock entries grouped together, making navigation much easier."

## Slide 15: Conclusion & Future Work (30 seconds)

- "DSR-RL successfully improves clustering without requiring labeled training data, handles both Arabic and English ambiguous queries, and adapts dynamically to user preferences."
- "Future work includes multimodal integration with images and audio, and mobile optimization for on-device processing."
- "Thank you for your attention. We're ready for questions."

---

# Key Technical Points to Remember:
- **BERTopic**: Uses UMAP for dimensionality reduction + HDBSCAN for clustering + c-TF-IDF for topic representation
- **Sentence-BERT**: Creates dense 384-dimensional vectors for semantic similarity
- **Q-Learning**: Tabular RL with discrete state space and ε-greedy exploration
- **Arabic NLP**: Diacritic removal, normalization, and morphological processing
- **Performance**: 13% purity improvement, 23% ARI improvement, <1s latency