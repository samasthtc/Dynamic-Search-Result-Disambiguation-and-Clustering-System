"""
Data templates for search result simulation
Contains predefined data for ambiguous queries and domain information
"""

# Predefined ambiguous queries with realistic results
AMBIGUOUS_QUERIES = {
    'jackson': {
        'ambiguity_level': 0.92,
        'entity_types': ['person', 'location'],
        'results': [
            {
                'title': 'Michael Jackson - King of Pop Biography',
                'snippet': 'Michael Joseph Jackson was an American singer, songwriter, and dancer. Dubbed the "King of Pop", he is regarded as one of the most significant cultural figures of the 20th century.',
                'url': 'https://www.michaeljackson.com/biography',
                'domain': 'michaeljackson.com',
                'category': 'person_musician',
                'authority': 0.95,
                'relevance': 0.98,
                'publish_date': '2023-06-15',
                'social_signals': {'shares': 45000, 'likes': 150000, 'comments': 8200}
            },
            {
                'title': 'Jackson, Mississippi - Capital City Guide',
                'snippet': 'Jackson is the capital and most populous city of Mississippi. The city serves as the anchor for the Jackson metropolitan statistical area.',
                'url': 'https://www.jacksonms.gov',
                'domain': 'jacksonms.gov',
                'category': 'location',
                'authority': 0.88,
                'relevance': 0.89,
                'publish_date': '2023-09-20',
                'social_signals': {'shares': 1200, 'likes': 5800, 'comments': 320}
            },
            {
                'title': 'Andrew Jackson - 7th President of the United States',
                'snippet': 'Andrew Jackson was the seventh President of the United States from 1829 to 1837. Before being elected president, Jackson gained fame as a general.',
                'url': 'https://www.whitehouse.gov/about-the-white-house/presidents/andrew-jackson',
                'domain': 'whitehouse.gov',
                'category': 'person_politician',
                'authority': 0.98,
                'relevance': 0.94,
                'publish_date': '2023-01-10',
                'social_signals': {'shares': 8500, 'likes': 25000, 'comments': 1850}
            },
            {
                'title': 'Jackson Pollock - Abstract Expressionist Artist',
                'snippet': 'Paul Jackson Pollock was an American painter and a major figure in the abstract expressionist movement. He was widely noticed for his technique of pouring paint.',
                'url': 'https://www.jackson-pollock.org',
                'domain': 'jackson-pollock.org',
                'category': 'person_artist',
                'authority': 0.85,
                'relevance': 0.86,
                'publish_date': '2023-03-22',
                'social_signals': {'shares': 3200, 'likes': 12500, 'comments': 890}
            },
            {
                'title': 'Janet Jackson - Music Icon and Performer',
                'snippet': 'Janet Damita Jackson is an American singer, songwriter, actress, and dancer. She is noted for her innovative, socially conscious records.',
                'url': 'https://www.janetjackson.com',
                'domain': 'janetjackson.com',
                'category': 'person_musician',
                'authority': 0.88,
                'relevance': 0.84,
                'publish_date': '2023-08-18',
                'social_signals': {'shares': 12000, 'likes': 58000, 'comments': 4200}
            }
        ]
    },
    
    'apple': {
        'ambiguity_level': 0.88,
        'entity_types': ['company', 'food'],
        'results': [
            {
                'title': 'Apple Inc. - Official Website',
                'snippet': 'Discover the innovative world of Apple and shop everything iPhone, iPad, Apple Watch, Mac, and Apple TV, plus explore accessories and expert device support.',
                'url': 'https://www.apple.com',
                'domain': 'apple.com',
                'category': 'company',
                'authority': 0.99,
                'relevance': 0.99,
                'publish_date': '2023-12-01',
                'social_signals': {'shares': 85000, 'likes': 320000, 'comments': 15800}
            },
            {
                'title': 'Apple Fruit - Nutrition and Health Benefits',
                'snippet': 'Apples are among the most nutritious fruits available. They are rich in fiber, vitamins, and antioxidants. Regular consumption may reduce disease risk.',
                'url': 'https://www.healthline.com/nutrition/10-health-benefits-of-apples',
                'domain': 'healthline.com',
                'category': 'food_health',
                'authority': 0.86,
                'relevance': 0.82,
                'publish_date': '2023-09-14',
                'social_signals': {'shares': 15000, 'likes': 45000, 'comments': 2800}
            },
            {
                'title': 'iPhone 15 Pro - Latest Apple Smartphone',
                'snippet': 'iPhone 15 Pro features a titanium design, Action button, Advanced camera system, and A17 Pro chip. Experience the most powerful iPhone ever.',
                'url': 'https://www.apple.com/iphone-15-pro',
                'domain': 'apple.com',
                'category': 'product_tech',
                'authority': 0.98,
                'relevance': 0.95,
                'publish_date': '2023-09-12',
                'social_signals': {'shares': 65000, 'likes': 180000, 'comments': 12500}
            },
            {
                'title': 'Apple Stock Analysis (AAPL) - Market Performance',
                'snippet': 'Apple Inc. stock performance analysis including price trends, financial metrics, earnings reports, and analyst recommendations for NASDAQ: AAPL.',
                'url': 'https://finance.yahoo.com/quote/AAPL',
                'domain': 'finance.yahoo.com',
                'category': 'finance',
                'authority': 0.92,
                'relevance': 0.88,
                'publish_date': '2023-12-05',
                'social_signals': {'shares': 25000, 'likes': 68000, 'comments': 5200}
            }
        ]
    },
    
    'python': {
        'ambiguity_level': 0.85,
        'entity_types': ['technology', 'animal'],
        'results': [
            {
                'title': 'Python Programming Language - Official Site',
                'snippet': 'Python is a programming language that lets you work quickly and integrate systems more effectively. Learn about Python syntax, libraries, and applications.',
                'url': 'https://www.python.org',
                'domain': 'python.org',
                'category': 'programming',
                'authority': 0.98,
                'relevance': 0.97,
                'publish_date': '2023-11-20',
                'social_signals': {'shares': 45000, 'likes': 125000, 'comments': 8500}
            },
            {
                'title': 'Python Snakes - Species and Behavior Guide',
                'snippet': 'Pythons are large, non-venomous snakes found in Africa, Asia, and Australia. They kill prey by constriction and are among the largest snakes.',
                'url': 'https://www.nationalgeographic.com/animals/reptiles/facts/pythons',
                'domain': 'nationalgeographic.com',
                'category': 'animal',
                'authority': 0.94,
                'relevance': 0.86,
                'publish_date': '2023-07-28',
                'social_signals': {'shares': 12000, 'likes': 42000, 'comments': 2100}
            },
            {
                'title': 'Learn Python Programming - Complete Tutorial',
                'snippet': 'Comprehensive Python tutorial covering basics to advanced concepts. Interactive examples, exercises, and projects for beginners and experienced programmers.',
                'url': 'https://www.w3schools.com/python',
                'domain': 'w3schools.com',
                'category': 'programming_education',
                'authority': 0.89,
                'relevance': 0.93,
                'publish_date': '2023-10-15',
                'social_signals': {'shares': 28000, 'likes': 95000, 'comments': 6800}
            },
            {
                'title': 'Ball Python Care Guide - Pet Snake Ownership',
                'snippet': 'Complete guide to ball python care including habitat setup, feeding, temperature requirements, health monitoring, and breeding information.',
                'url': 'https://www.reptilecentre.com/ball-python-care',
                'domain': 'reptilecentre.com',
                'category': 'animal_care',
                'authority': 0.72,
                'relevance': 0.74,
                'publish_date': '2023-08-30',
                'social_signals': {'shares': 3800, 'likes': 16500, 'comments': 1200}
            }
        ]
    },
    
    'mercury': {
        'ambiguity_level': 0.87,
        'entity_types': ['celestial_body', 'chemical_element', 'person'],
        'results': [
            {
                'title': 'Mercury Planet - NASA Solar System Exploration',
                'snippet': 'Mercury is the smallest planet in our solar system and the one closest to the Sun. It has extreme temperature variations and virtually no atmosphere.',
                'url': 'https://solarsystem.nasa.gov/planets/mercury',
                'domain': 'nasa.gov',
                'category': 'astronomy',
                'authority': 0.98,
                'relevance': 0.92,
                'publish_date': '2023-09-05',
                'social_signals': {'shares': 18000, 'likes': 65000, 'comments': 3500}
            },
            {
                'title': 'Freddie Mercury - Queen Lead Singer Biography',
                'snippet': 'Freddie Mercury was a British singer, songwriter, and record producer, best known as the lead vocalist of the rock band Queen.',
                'url': 'https://www.freddiemercury.com',
                'domain': 'freddiemercury.com',
                'category': 'person_musician',
                'authority': 0.90,
                'relevance': 0.89,
                'publish_date': '2023-11-24',
                'social_signals': {'shares': 22000, 'likes': 95000, 'comments': 6200}
            },
            {
                'title': 'Mercury Chemical Element - Properties and Uses',
                'snippet': 'Mercury is a chemical element with symbol Hg and atomic number 80. It is the only metallic element that is liquid at standard conditions.',
                'url': 'https://www.britannica.com/science/mercury-chemical-element',
                'domain': 'britannica.com',
                'category': 'chemistry',
                'authority': 0.93,
                'relevance': 0.85,
                'publish_date': '2023-06-18',
                'social_signals': {'shares': 5200, 'likes': 18500, 'comments': 980}
            }
        ]
    }
}

# Domain authority scores for realistic ranking
DOMAIN_AUTHORITIES = {
    # Tech companies
    'google.com': 0.99,
    'apple.com': 0.99,
    'microsoft.com': 0.98,
    'amazon.com': 0.98,
    'facebook.com': 0.97,
    
    # Government and institutions
    'nasa.gov': 0.98,
    'whitehouse.gov': 0.98,
    'wikipedia.org': 0.96,
    
    # News and media
    'bbc.com': 0.94,
    'cnn.com': 0.92,
    'nytimes.com': 0.95,
    'nationalgeographic.com': 0.94,
    
    # Educational
    'w3schools.com': 0.89,
    'python.org': 0.98,
    'britannica.com': 0.93,
    
    # Health
    'healthline.com': 0.86,
    'mayoclinic.org': 0.95,
    
    # Finance
    'finance.yahoo.com': 0.92,
    
    # Specialized
    'michaeljackson.com': 0.95,
    'freddiemercury.com': 0.90,
    'jackson-pollock.org': 0.85,
    'reptilecentre.com': 0.72,
    'jacksonms.gov': 0.88,
    'janetjackson.com': 0.88,
    
    # Default
    'default': 0.50
}

# Content type characteristics
CONTENT_TYPES = {
    'biography': {
        'engagement_factor': 1.2,
        'reading_time_multiplier': 1.0,
        'typical_length': (1000, 3000),
        'authority_boost': 0.1
    },
    'news': {
        'engagement_factor': 1.8,
        'reading_time_multiplier': 0.9,
        'typical_length': (500, 1500),
        'authority_boost': 0.15
    },
    'tutorial': {
        'engagement_factor': 1.5,
        'reading_time_multiplier': 1.2,
        'typical_length': (2000, 8000),
        'authority_boost': 0.05
    },
    'product': {
        'engagement_factor': 1.0,
        'reading_time_multiplier': 0.8,
        'typical_length': (300, 1000),
        'authority_boost': 0.2
    },
    'government': {
        'engagement_factor': 0.6,
        'reading_time_multiplier': 1.1,
        'typical_length': (800, 2500),
        'authority_boost': 0.3
    },
    'health': {
        'engagement_factor': 1.3,
        'reading_time_multiplier': 1.1,
        'typical_length': (1200, 4000),
        'authority_boost': 0.2
    },
    'programming': {
        'engagement_factor': 1.4,
        'reading_time_multiplier': 1.3,
        'typical_length': (1500, 5000),
        'authority_boost': 0.1
    }
}

# User behavior profiles
USER_PROFILES = {
    'novice': {
        'attention_span': 2.0,
        'results_examined': 0.4,
        'click_threshold': 0.6,
        'brand_preference': 0.3,
        'satisfaction_threshold': 0.5
    },
    'average': {
        'attention_span': 3.0,
        'results_examined': 0.7,
        'click_threshold': 0.5,
        'brand_preference': 0.2,
        'satisfaction_threshold': 0.65
    },
    'expert': {
        'attention_span': 2.5,
        'results_examined': 0.9,
        'click_threshold': 0.4,
        'brand_preference': 0.1,
        'satisfaction_threshold': 0.8
    },
    'researcher': {
        'attention_span': 4.0,
        'results_examined': 0.95,
        'click_threshold': 0.3,
        'brand_preference': 0.05,
        'satisfaction_threshold': 0.85
    }
}

# Search intent modifiers
INTENT_MODIFIERS = {
    'informational': {
        'patience_factor': 1.2,
        'depth_factor': 1.0,
        'speed_factor': 1.0
    },
    'navigational': {
        'patience_factor': 0.8,
        'depth_factor': 0.6,
        'speed_factor': 1.4
    },
    'transactional': {
        'patience_factor': 1.0,
        'depth_factor': 0.8,
        'speed_factor': 1.2
    },
    'commercial': {
        'patience_factor': 1.1,
        'depth_factor': 1.2,
        'speed_factor': 0.9
    }
}
