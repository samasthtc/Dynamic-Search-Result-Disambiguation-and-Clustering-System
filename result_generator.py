"""
Search result generation module
Handles creation of realistic search results from templates and generic generation
"""

import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from .data_templates import AMBIGUOUS_QUERIES, DOMAIN_AUTHORITIES, CONTENT_TYPES

logger = logging.getLogger(__name__)


class ResultGenerator:
    """
    Generates realistic search results for queries.
    Handles both predefined templates and generic result creation.
    """
    
    def __init__(self):
        self.base_relevance_variance = 0.1
        self.authority_weight = 0.2
        self.freshness_weight = 0.15
        
    def generate_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Generate search results for a given query.
        
        Args:
            query: Search query
            num_results: Number of results to generate
            
        Returns:
            List of search result dictionaries
        """
        query_lower = query.lower().strip()
        
        if query_lower in AMBIGUOUS_QUERIES:
            results = self._generate_from_templates(query_lower, num_results)
        else:
            results = self._generate_generic_results(query, num_results)
        
        # Enhance all results with metadata
        enhanced_results = []
        for result in results:
            enhanced = self._enhance_result(result, query)
            enhanced_results.append(enhanced)
        
        # Sort by relevance score
        enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return enhanced_results[:num_results]
    
    def _generate_from_templates(self, query: str, num_results: int) -> List[Dict]:
        """Generate results from predefined templates"""
        query_data = AMBIGUOUS_QUERIES[query]
        templates = query_data['results'].copy()
        
        results = []
        
        # Use all predefined templates first
        for template in templates:
            result = self._create_from_template(template)
            results.append(result)
        
        # Generate additional results if needed
        while len(results) < num_results:
            if len(results) < len(templates) * 2:
                # Create variations of existing templates
                base_template = random.choice(templates)
                variation = self._create_template_variation(base_template, query)
                results.append(variation)
            else:
                # Create generic results
                generic = self._create_generic_result(query, len(results))
                results.append(generic)
        
        return results
    
    def _create_from_template(self, template: Dict) -> Dict:
        """Create a complete result from a template"""
        domain = template['domain']
        authority = DOMAIN_AUTHORITIES.get(domain, 0.5)
        
        # Generate dates
        pub_date = datetime.strptime(template['publish_date'], '%Y-%m-%d')
        last_update = pub_date + timedelta(days=random.randint(0, 60))
        
        result = {
            'id': self._generate_id(template['url']),
            'title': template['title'],
            'snippet': template['snippet'],
            'url': template['url'],
            'domain': domain,
            'category': template['category'],
            'language': 'en',
            
            # Core scoring factors
            'base_relevance': template['relevance'],
            'authority_score': authority,
            'publish_date': template['publish_date'],
            'last_updated': last_update.strftime('%Y-%m-%d'),
            
            # Content metadata
            'word_count': self._estimate_word_count(template['snippet'], template['category']),
            'reading_time': self._calculate_reading_time(template['snippet'], template['category']),
            'has_images': random.choice([True, False]),
            'has_videos': self._has_videos(template['category']),
            
            # Technical factors
            'mobile_friendly': authority > 0.7,
            'ssl_enabled': authority > 0.6,
            'page_speed': random.uniform(0.4, 1.0),
            'accessibility_score': random.uniform(0.3, 0.9),
            
            # Engagement metrics
            'social_signals': template['social_signals'],
            'estimated_ctr': self._estimate_ctr(template),
            'bounce_rate': random.uniform(0.2, 0.8),
            'time_on_page': random.uniform(30, 300),
            
            # Quality indicators
            'freshness_score': self._calculate_freshness(template['publish_date']),
            'quality_score': self._calculate_quality(template, authority)
        }
        
        return result
    
    def _create_template_variation(self, base_template: Dict, query: str) -> Dict:
        """Create a variation of an existing template"""
        variation_prefixes = [
            'Complete Guide to', 'Everything About', 'Understanding',
            'Explore', 'Discover', 'Learn About'
        ]
        
        new_title = f"{random.choice(variation_prefixes)} {base_template['title']}"
        new_snippet = f"Comprehensive information about {query}. {base_template['snippet'][:80]}..."
        
        # Generate new domain and URL
        variation_domains = ['guide.com', 'info.org', 'learn.net', 'reference.co']
        new_domain = random.choice(variation_domains)
        new_url = f"https://www.{new_domain}/{query.replace(' ', '-')}"
        
        variation = base_template.copy()
        variation.update({
            'title': new_title,
            'snippet': new_snippet,
            'url': new_url,
            'domain': new_domain,
            'relevance': max(0.3, base_template['relevance'] - 0.2),
            'social_signals': {
                'shares': base_template['social_signals']['shares'] // 3,
                'likes': base_template['social_signals']['likes'] // 3,
                'comments': base_template['social_signals']['comments'] // 3
            }
        })
        
        return self._create_from_template(variation)
    
    def _generate_generic_results(self, query: str, num_results: int) -> List[Dict]:
        """Generate generic results for unknown queries"""
        results = []
        
        generic_categories = [
            'general', 'news', 'tutorial', 'product', 'company',
            'person', 'location', 'review', 'forum', 'video'
        ]
        
        generic_domains = [
            'wikipedia.org', 'britannica.com', 'example.com', 'info.org',
            'guide.net', 'learn.co', 'reference.com', 'knowledge.edu'
        ]
        
        for i in range(num_results):
            category = random.choice(generic_categories)
            domain = random.choice(generic_domains)
            result = self._create_generic_result(query, i, category, domain)
            results.append(result)
        
        return results
    
    def _create_generic_result(self, query: str, index: int, 
                              category: str = None, domain: str = None) -> Dict:
        """Create a single generic search result"""
        if not category:
            category = 'general'
        if not domain:
            domain = 'example.com'
        
        # Generate contextual content
        title_templates = {
            'general': f"{query.title()} - Complete Information Guide",
            'news': f"Latest {query.title()} News and Updates",
            'tutorial': f"How to Learn {query.title()} - Step by Step Guide",
            'product': f"{query.title()} Products and Reviews",
            'company': f"{query.title()} Company Information",
            'person': f"{query.title()} Biography and Information",
            'location': f"{query.title()} Travel and Location Guide",
            'review': f"{query.title()} Reviews and Analysis",
            'forum': f"{query.title()} Community Discussion",
            'video': f"{query.title()} Videos and Tutorials"
        }
        
        snippet_templates = {
            'general': f"Comprehensive information about {query} including definitions, characteristics, and important details.",
            'news': f"Stay updated with the latest news and developments about {query} from reliable sources.",
            'tutorial': f"Learn everything about {query} with step-by-step instructions and practical examples.",
            'product': f"Discover the best {query} products, compare features, prices, and read user reviews.",
            'company': f"Official {query} company information including services, contact details, and business overview.",
            'person': f"Biography and detailed information about {query} including achievements and background.",
            'location': f"Complete travel guide for {query} including attractions, accommodations, and local information.",
            'review': f"Honest reviews and detailed analysis of {query} with pros, cons, and recommendations.",
            'forum': f"Join the {query} community discussion, ask questions, and share experiences.",
            'video': f"Watch {query} videos, tutorials, and educational content from experts."
        }
        
        title = title_templates.get(category, f"{query.title()} Information")
        snippet = snippet_templates.get(category, f"Information about {query}")
        
        return {
            'id': self._generate_id(f"https://www.{domain}/{query}"),
            'title': title,
            'snippet': snippet,
            'url': f"https://www.{domain}/{query.replace(' ', '-').lower()}",
            'domain': domain,
            'category': category,
            'language': 'en',
            'base_relevance': random.uniform(0.4, 0.8),
            'authority_score': DOMAIN_AUTHORITIES.get(domain, 0.5),
            'publish_date': self._generate_date(),
            'last_updated': self._generate_recent_date(),
            'word_count': random.randint(300, 2000),
            'reading_time': random.randint(2, 8),
            'has_images': random.choice([True, False]),
            'has_videos': random.choice([True, False]),
            'mobile_friendly': random.choice([True, True, False]),
            'ssl_enabled': random.choice([True, True, False]),
            'page_speed': random.uniform(0.3, 0.9),
            'social_signals': self._generate_social_signals(),
            'estimated_ctr': random.uniform(0.02, 0.15),
            'bounce_rate': random.uniform(0.3, 0.8),
            'time_on_page': random.uniform(30, 180)
        }
    
    def _enhance_result(self, result: Dict, query: str) -> Dict:
        """Enhance result with computed scores and metadata"""
        # Calculate final relevance score
        base_score = result.get('base_relevance', 0.5)
        authority_boost = result.get('authority_score', 0.5) * self.authority_weight
        freshness_boost = result.get('freshness_score', 0.5) * self.freshness_weight
        
        # Query match score
        query_match = self._calculate_query_match(result['title'] + ' ' + result['snippet'], query)
        
        # Social signals boost
        social_boost = self._normalize_social_signals(result.get('social_signals', {})) * 0.1
        
        # Technical score
        technical_score = self._calculate_technical_score(result)
        
        # Calculate final score
        final_score = (
            base_score + 
            authority_boost + 
            freshness_boost + 
            query_match * 0.2 + 
            social_boost + 
            technical_score * 0.05
        )
        
        # Add realistic variance
        variance = random.uniform(-self.base_relevance_variance, self.base_relevance_variance)
        final_score = max(0.0, min(1.0, final_score + variance))
        
        # Add computed fields
        result['query_match_score'] = query_match
        result['technical_score'] = technical_score
        result['final_score'] = final_score
        
        # Add predictions
        result['click_probability'] = self._predict_click_probability(result)
        result['dwell_time_prediction'] = self._predict_dwell_time(result)
        result['satisfaction_prediction'] = self._predict_satisfaction(result)
        
        return result
    
    def _generate_id(self, url: str) -> str:
        """Generate consistent ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def _generate_date(self, days_back: int = 730) -> str:
        """Generate random publication date"""
        days_ago = random.randint(1, days_back)
        date = datetime.now() - timedelta(days=days_ago)
        return date.strftime('%Y-%m-%d')
    
    def _generate_recent_date(self, days_back: int = 180) -> str:
        """Generate recent update date"""
        days_ago = random.randint(0, days_back)
        date = datetime.now() - timedelta(days=days_ago)
        return date.strftime('%Y-%m-%d')
    
    def _estimate_word_count(self, snippet: str, category: str) -> int:
        """Estimate full article word count from snippet"""
        snippet_words = len(snippet.split())
        
        content_info = CONTENT_TYPES.get(category, CONTENT_TYPES.get('general', {}))
        length_range = content_info.get('typical_length', (500, 1500))
        
        # Generate word count within typical range for content type
        return random.randint(length_range[0], length_range[1])
    
    def _calculate_reading_time(self, snippet: str, category: str) -> int:
        """Calculate estimated reading time"""
        word_count = self._estimate_word_count(snippet, category)
        words_per_minute = 200
        return max(1, word_count // words_per_minute)
    
    def _has_videos(self, category: str) -> bool:
        """Determine if content likely has videos"""
        video_likelihood = {
            'tutorial': 0.7,
            'product': 0.6,
            'news': 0.4,
            'person': 0.3,
            'programming': 0.8,
            'general': 0.2
        }
        
        probability = video_likelihood.get(category, 0.3)
        return random.random() < probability
    
    def _estimate_ctr(self, template: Dict) -> float:
        """Estimate click-through rate"""
        base_ctr = 0.1
        
        # Authority boost
        authority_boost = template.get('authority', 0.5) * 0.1
        
        # Title attractiveness
        title = template['title'].lower()
        attractive_words = ['ultimate', 'complete', 'best', 'guide', 'official']
        title_boost = sum(0.02 for word in attractive_words if word in title)
        
        return min(0.5, base_ctr + authority_boost + title_boost)
    
    def _calculate_freshness(self, publish_date: str) -> float:
        """Calculate freshness score based on publication date"""
        try:
            pub_date = datetime.strptime(publish_date, '%Y-%m-%d')
            days_old = (datetime.now() - pub_date).days
            
            # Exponential decay
            import math
            freshness = math.exp(-days_old / 365.25)
            return max(0.1, min(1.0, freshness))
        except:
            return 0.5
    
    def _calculate_quality(self, template: Dict, authority: float) -> float:
        """Calculate overall quality score"""
        quality_factors = [
            authority,
            template.get('relevance', 0.5),
            min(1.0, len(template['snippet']) / 200),  # Content length indicator
            1.0 if template.get('social_signals', {}).get('shares', 0) > 1000 else 0.5
        ]
        
        return sum(quality_factors) / len(quality_factors)
    
    def _calculate_query_match(self, text: str, query: str) -> float:
        """Calculate how well text matches query"""
        text_lower = text.lower()
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Exact match bonus
        exact_match = 0.5 if query_lower in text_lower else 0.0
        
        # Word coverage
        word_matches = sum(1 for word in query_words if word in text_lower)
        word_coverage = (word_matches / len(query_words)) * 0.3 if query_words else 0.0
        
        # Position bonus
        position_bonus = 0.0
        if query_lower in text_lower:
            position = text_lower.find(query_lower)
            position_bonus = max(0, (1 - position / len(text_lower))) * 0.2
        
        return min(1.0, exact_match + word_coverage + position_bonus)
    
    def _normalize_social_signals(self, social_signals: Dict) -> float:
        """Normalize social signals to 0-1 scale"""
        if not social_signals:
            return 0.0
        
        total_engagement = sum(social_signals.values())
        if total_engagement == 0:
            return 0.0
        
        # Logarithmic normalization
        import math
        normalized = math.log(total_engagement + 1) / math.log(100000)
        return min(1.0, normalized)
    
    def _calculate_technical_score(self, result: Dict) -> float:
        """Calculate technical SEO score"""
        factors = [
            1.0 if result.get('mobile_friendly', False) else 0.0,
            1.0 if result.get('ssl_enabled', False) else 0.0,
            result.get('page_speed', 0.5),
            result.get('accessibility_score', 0.5)
        ]
        
        return sum(factors) / len(factors)
    
    def _predict_click_probability(self, result: Dict) -> float:
        """Predict probability of user clicking this result"""
        factors = [
            result.get('final_score', 0.5),
            result.get('authority_score', 0.5) * 0.5,
            result.get('query_match_score', 0.5) * 0.8,
            result.get('technical_score', 0.5) * 0.3
        ]
        
        base_probability = sum(factors) / len(factors)
        return max(0.01, min(0.8, base_probability))
    
    def _predict_dwell_time(self, result: Dict) -> float:
        """Predict time user will spend on page (seconds)"""
        base_time = result.get('time_on_page', 60)
        
        # Adjust based on content type and quality
        quality_multiplier = result.get('quality_score', 0.5) + 0.5
        reading_time_factor = result.get('reading_time', 3) * 20  # Convert minutes to seconds
        
        predicted_time = (base_time * quality_multiplier + reading_time_factor) / 2
        return max(10, min(600, predicted_time))  # Between 10 seconds and 10 minutes
    
    def _predict_satisfaction(self, result: Dict) -> float:
        """Predict user satisfaction probability"""
        factors = [
            result.get('final_score', 0.5) * 0.4,
            result.get('authority_score', 0.5) * 0.3,
            result.get('technical_score', 0.5) * 0.2,
            result.get('query_match_score', 0.5) * 0.1
        ]
        
        satisfaction = sum(factors)
        return max(0.1, min(1.0, satisfaction))
    
    def _generate_social_signals(self, base_engagement: int = None) -> Dict[str, int]:
        """Generate realistic social media engagement"""
        if base_engagement is None:
            base_engagement = random.randint(10, 10000)
        
        return {
            'shares': random.randint(base_engagement // 20, base_engagement),
            'likes': random.randint(base_engagement, base_engagement * 3),
            'comments': random.randint(base_engagement // 50, base_engagement // 10)
        }
