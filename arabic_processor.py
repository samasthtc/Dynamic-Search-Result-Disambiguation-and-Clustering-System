import re
import unicodedata
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ArabicTextProcessor:
    """
    Advanced Arabic text processor for handling morphological richness,
    lexical ambiguity, and orthographic variance in Arabic search queries.
    """
    
    def __init__(self):
        # Arabic character mappings and normalization rules
        self.arabic_normalizations = self._load_arabic_normalizations()
        self.stop_words = self._load_arabic_stop_words()
        self.morphological_patterns = self._load_morphological_patterns()
        
        # Arabic script ranges
        self.arabic_range = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
        
        logger.info("Arabic Text Processor initialized")

    def _load_arabic_normalizations(self) -> Dict[str, str]:
        """Load Arabic character normalization mappings"""
        return {
            # Alef variations
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
            # Yeh variations  
            'ي': 'ى', 'ئ': 'ى', 'ؤ': 'و',
            # Teh marbuta
            'ة': 'ه',
            # Remove diacritics (Tashkeel)
            'َ': '', 'ُ': '', 'ِ': '', 'ً': '', 'ٌ': '', 'ٍ': '',
            'ْ': '', 'ّ': '', 'ـ': '',
            # Remove Tatweel (elongation)
            'ـ': '',
        }

    def _load_arabic_stop_words(self) -> List[str]:
        """Load common Arabic stop words"""
        return [
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
            'التي', 'الذي', 'التي', 'اللذان', 'اللتان', 'اللذين', 'اللتين',
            'هو', 'هي', 'هم', 'هن', 'أنت', 'أنتم', 'أنتن', 'أنا', 'نحن',
            'لا', 'ما', 'لم', 'لن', 'إن', 'أن', 'كان', 'كانت', 'يكون', 'تكون',
            'بعد', 'قبل', 'أثناء', 'خلال', 'عند', 'لدى', 'حول', 'بين', 'ضد',
            'رغم', 'بسبب', 'نتيجة', 'بدلا', 'عوضا', 'حيث', 'بينما', 'عندما',
            'كل', 'بعض', 'جميع', 'معظم', 'أكثر', 'أقل', 'نفس', 'ذات', 'غير',
            'سوى', 'عدا', 'خلا', 'ماعدا', 'ماخلا', 'فقط', 'أيضا', 'كذلك',
            'إذ', 'إذا', 'لو', 'لولا', 'لوما', 'كلما', 'مهما', 'أينما', 'حيثما'
        ]

    def _load_morphological_patterns(self) -> List[Dict[str, str]]:
        """Load Arabic morphological patterns for root extraction"""
        return [
            # Common trilateral root patterns (فعل)
            {'pattern': r'^م(..)(.)(.)$', 'root_positions': [1, 2, 3]},  # مفعل
            {'pattern': r'^(..)ا(.)(.)$', 'root_positions': [1, 3, 4]},   # فاعل
            {'pattern': r'^(..)و(.)(.)$', 'root_positions': [1, 3, 4]},   # فوعل
            {'pattern': r'^(..)ي(.)(.)$', 'root_positions': [1, 3, 4]},   # فيعل
            {'pattern': r'^ت(..)(.)(.)$', 'root_positions': [1, 2, 3]},   # تفعل
            {'pattern': r'^ا(..)(.)(.)$', 'root_positions': [1, 2, 3]},   # افعل
            {'pattern': r'^(..)(.)(.)ة$', 'root_positions': [1, 2, 3]},   # فعلة
            {'pattern': r'^(..)(.)(.)ان$', 'root_positions': [1, 2, 3]},  # فعلان
            {'pattern': r'^(..)(.)(.)ين$', 'root_positions': [1, 2, 3]},  # فعلين
            # Quadrilateral patterns (فعلل)
            {'pattern': r'^(..)(.)(.)(..)$', 'root_positions': [1, 2, 3, 4]}, # فعلل
        ]

    def preprocess_text(self, text: str) -> str:
        """
        Comprehensive Arabic text preprocessing including normalization,
        morphological analysis, and orthographic variance handling.
        """
        if not text or not self._is_arabic_text(text):
            return text
        
        # Step 1: Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Step 2: Character-level normalization
        text = self._normalize_arabic_chars(text)
        
        # Step 3: Remove diacritics and elongations
        text = self._remove_diacritics(text)
        
        # Step 4: Handle orthographic variants
        text = self._handle_orthographic_variants(text)
        
        # Step 5: Normalize spaces and punctuation
        text = self._normalize_punctuation(text)
        
        return text.strip()

    def tokenize_arabic(self, text: str) -> List[str]:
        """
        Advanced Arabic tokenization handling morphological complexity.
        """
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Basic tokenization by spaces and punctuation
        tokens = re.findall(r'[^\s\u060C\u061B\u061F\u0640]+', text)
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply morphological analysis
        processed_tokens = []
        for token in tokens:
            # Try to extract root or stem
            root = self._extract_arabic_root(token)
            if root and len(root) >= 3:
                processed_tokens.append(root)
            else:
                processed_tokens.append(token)
        
        return processed_tokens

    def extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """
        Extract semantic features from Arabic text for better disambiguation.
        """
        features = {
            'original_text': text,
            'preprocessed_text': self.preprocess_text(text),
            'word_count': 0,
            'character_count': 0,
            'has_diacritics': False,
            'script_complexity': 0.0,
            'morphological_richness': 0.0,
            'semantic_density': 0.0
        }
        
        if not self._is_arabic_text(text):
            return features
        
        # Basic metrics
        tokens = self.tokenize_arabic(text)
        features['word_count'] = len(tokens)
        features['character_count'] = len(text)
        
        # Diacritics detection
        features['has_diacritics'] = bool(re.search(r'[ًَُِ ٌٍّْ]', text))
        
        # Script complexity (variety of character forms)
        unique_chars = set(re.findall(self.arabic_range, text))
        features['script_complexity'] = len(unique_chars) / max(1, len(text))
        
        # Morphological richness (affixation patterns)
        features['morphological_richness'] = self._calculate_morphological_richness(tokens)
        
        # Semantic density (content word ratio)
        features['semantic_density'] = self._calculate_semantic_density(tokens)
        
        return features

    def handle_query_ambiguity(self, query: str) -> Dict[str, Any]:
        """
        Analyze and handle ambiguity in Arabic search queries.
        """
        ambiguity_analysis = {
            'original_query': query,
            'ambiguity_level': 0.0,
            'ambiguity_sources': [],
            'disambiguation_suggestions': [],
            'morphological_variants': [],
            'semantic_expansions': []
        }
        
        if not self._is_arabic_text(query):
            return ambiguity_analysis
        
        tokens = self.tokenize_arabic(query)
        
        # Analyze each token for ambiguity
        for token in tokens:
            # Check for morphological ambiguity
            variants = self._generate_morphological_variants(token)
            if variants:
                ambiguity_analysis['morphological_variants'].extend(variants)
                ambiguity_analysis['ambiguity_sources'].append('morphological')
            
            # Check for semantic ambiguity (homonyms)
            if self._is_potentially_ambiguous(token):
                ambiguity_analysis['ambiguity_sources'].append('semantic')
                ambiguity_analysis['semantic_expansions'].extend(
                    self._generate_semantic_expansions(token)
                )
        
        # Calculate overall ambiguity level
        ambiguity_factors = len(set(ambiguity_analysis['ambiguity_sources']))
        variant_count = len(ambiguity_analysis['morphological_variants'])
        expansion_count = len(ambiguity_analysis['semantic_expansions'])
        
        ambiguity_analysis['ambiguity_level'] = min(1.0, 
            (ambiguity_factors * 0.3 + variant_count * 0.05 + expansion_count * 0.02))
        
        # Generate disambiguation suggestions
        ambiguity_analysis['disambiguation_suggestions'] = self._generate_disambiguation_suggestions(query, tokens)
        
        return ambiguity_analysis

    def _is_arabic_text(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        return bool(re.search(self.arabic_range, text))

    def _normalize_arabic_chars(self, text: str) -> str:
        """Normalize Arabic character variations"""
        for original, normalized in self.arabic_normalizations.items():
            text = text.replace(original, normalized)
        return text

    def _remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics (Tashkeel)"""
        diacritics = r'[ًَُِ ٌٍّْ]'
        return re.sub(diacritics, '', text)

    def _handle_orthographic_variants(self, text: str) -> str:
        """Handle common orthographic variants in Arabic"""
        # Handle common spelling variations
        variants = {
            'اللة': 'الله',  # Allah spelling variant
            'إن شاء الله': 'انشاءالله',  # InshAllah variants
            'ابراهيم': 'إبراهيم',  # Ibrahim variants
        }
        
        for variant, standard in variants.items():
            text = text.replace(variant, standard)
        
        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize Arabic punctuation and spaces"""
        # Replace Arabic punctuation with standard forms
        text = re.sub(r'[،؍؎؏ؐؑؒؓؔؕؖؗ؛]', '،', text)  # Arabic comma variants
        text = re.sub(r'[؞؟]', '؟', text)  # Arabic question mark variants
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def _extract_arabic_root(self, word: str) -> str:
        """Extract Arabic root using morphological patterns"""
        if len(word) < 3:
            return word
        
        # Try each morphological pattern
        for pattern_info in self.morphological_patterns:
            pattern = pattern_info['pattern']
            root_positions = pattern_info['root_positions']
            
            match = re.match(pattern, word)
            if match:
                groups = match.groups()
                try:
                    root_chars = [groups[pos-1] for pos in root_positions if pos <= len(groups)]
                    root = ''.join(root_chars)
                    if len(root) >= 3:
                        return root
                except IndexError:
                    continue
        
        # If no pattern matches, return first 3 characters as approximation
        return word[:3] if len(word) >= 3 else word

    def _calculate_morphological_richness(self, tokens: List[str]) -> float:
        """Calculate morphological richness of token list"""
        if not tokens:
            return 0.0
        
        # Count tokens with potential morphological complexity
        complex_tokens = 0
        for token in tokens:
            if len(token) > 4:  # Longer words likely have more morphology
                complex_tokens += 1
            if self._has_morphological_markers(token):
                complex_tokens += 1
        
        return complex_tokens / len(tokens)

    def _has_morphological_markers(self, token: str) -> bool:
        """Check if token has morphological markers"""
        # Check for common prefixes and suffixes
        prefixes = ['ال', 'و', 'ف', 'ب', 'ل', 'ك', 'م', 'ت', 'ي', 'ن']
        suffixes = ['ة', 'ان', 'ين', 'ون', 'ها', 'هم', 'هن', 'كم', 'كن', 'تم', 'تن']
        
        for prefix in prefixes:
            if token.startswith(prefix) and len(token) > len(prefix) + 2:
                return True
        
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return True
        
        return False

    def _calculate_semantic_density(self, tokens: List[str]) -> float:
        """Calculate semantic density (ratio of content words)"""
        if not tokens:
            return 0.0
        
        content_words = 0
        for token in tokens:
            if token not in self.stop_words and len(token) >= 3:
                content_words += 1
        
        return content_words / len(tokens)

    def _generate_morphological_variants(self, token: str) -> List[str]:
        """Generate morphological variants of a token"""
        variants = [token]  # Include original
        
        if len(token) < 3:
            return variants
        
        # Generate variants by adding/removing common affixes
        prefixes = ['ال', 'و', 'ف', 'ب', 'ل', 'ك']
        suffixes = ['ة', 'ان', 'ين', 'ون', 'ها']
        
        # Remove prefixes if present
        for prefix in prefixes:
            if token.startswith(prefix) and len(token) > len(prefix) + 2:
                variant = token[len(prefix):]
                if variant not in variants:
                    variants.append(variant)
        
        # Remove suffixes if present
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                variant = token[:-len(suffix)]
                if variant not in variants:
                    variants.append(variant)
        
        # Add prefixes to root
        root = self._extract_arabic_root(token)
        if root != token:
            for prefix in prefixes[:3]:  # Limit to most common
                variant = prefix + root
                if variant not in variants:
                    variants.append(variant)
        
        return variants[:10]  # Limit to prevent explosion

    def _is_potentially_ambiguous(self, token: str) -> bool:
        """Check if a token is potentially semantically ambiguous"""
        # List of known ambiguous Arabic words
        ambiguous_words = {
            'عين': ['eye', 'spring', 'spy'],
            'بنك': ['bank_financial', 'bank_river'],
            'ورد': ['flowers', 'mentioned'],
            'سلم': ['peace', 'ladder', 'scale'],
            'نور': ['light', 'name'],
            'جوز': ['nuts', 'husband'],
            'قلم': ['pen', 'region'],
            'باب': ['door', 'chapter'],
            'كتاب': ['book', 'writing'],
            'مفتاح': ['key', 'opener']
        }
        
        return token in ambiguous_words

    def _generate_semantic_expansions(self, token: str) -> List[str]:
        """Generate semantic expansions for ambiguous tokens"""
        # Semantic expansion dictionary
        expansions = {
            'عين': ['عين_بصر', 'عين_ماء', 'عين_جاسوس'],
            'بنك': ['بنك_مالي', 'بنك_نهر'],
            'ورد': ['ورد_زهور', 'ورد_ذكر'],
            'سلم': ['سلم_أمان', 'سلم_درج', 'سلم_موسيقي'],
            'نور': ['نور_ضوء', 'نور_اسم'],
            'جوز': ['جوز_لوز', 'جوز_زوج'],
            'قلم': ['قلم_كتابة', 'قلم_منطقة'],
            'باب': ['باب_مدخل', 'باب_فصل'],
            'كتاب': ['كتاب_مؤلف', 'كتاب_كتابة'],
            'مفتاح': ['مفتاح_قفل', 'مفتاح_فتح']
        }
        
        return expansions.get(token, [])

    def _generate_disambiguation_suggestions(self, query: str, tokens: List[str]) -> List[str]:
        """Generate disambiguation suggestions for the query"""
        suggestions = []
        
        # Context-based suggestions
        context_hints = {
            'عين': {
                'medical': 'عين العضو البصري',
                'geography': 'عين الماء الطبيعية',
                'security': 'عين المراقبة والجاسوسية'
            },
            'بنك': {
                'finance': 'البنك المالي والمصرفي',
                'geography': 'بنك النهر الجانبي'
            },
            'سلم': {
                'peace': 'السلم والأمان',
                'tools': 'سلم التسلق',
                'music': 'السلم الموسيقي'
            }
        }
        
        for token in tokens:
            if token in context_hints:
                for context, suggestion in context_hints[token].items():
                    suggestions.append(suggestion)
        
        # Add general disambiguation suggestions
        if len(tokens) == 1:  # Single word queries are often ambiguous
            suggestions.extend([
                f'{query} + كلمة إضافية للتوضيح',
                f'أضف سياق لكلمة {query}',
                f'حدد المجال المقصود من {query}'
            ])
        
        return suggestions[:5]  # Limit suggestions

    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of Arabic query complexity for clustering optimization.
        """
        complexity_analysis = {
            'query': query,
            'complexity_score': 0.0,
            'complexity_factors': [],
            'processing_recommendations': [],
            'clustering_hints': []
        }
        
        if not self._is_arabic_text(query):
            complexity_analysis['complexity_score'] = 0.1  # Low complexity for non-Arabic
            return complexity_analysis
        
        tokens = self.tokenize_arabic(query)
        features = self.extract_semantic_features(query)
        ambiguity = self.handle_query_ambiguity(query)
        
        # Calculate complexity factors
        complexity_factors = []
        
        # 1. Morphological complexity
        if features['morphological_richness'] > 0.5:
            complexity_factors.append('high_morphological_richness')
            complexity_analysis['processing_recommendations'].append(
                'Apply advanced morphological analysis'
            )
        
        # 2. Semantic ambiguity
        if ambiguity['ambiguity_level'] > 0.3:
            complexity_factors.append('semantic_ambiguity')
            complexity_analysis['clustering_hints'].append(
                'Use disambiguation context for clustering'
            )
        
        # 3. Script complexity
        if features['script_complexity'] > 0.8:
            complexity_factors.append('complex_script_usage')
        
        # 4. Query length impact
        if features['word_count'] == 1:
            complexity_factors.append('single_word_ambiguity')
            complexity_analysis['clustering_hints'].append(
                'Single word queries need context-based clustering'
            )
        elif features['word_count'] > 5:
            complexity_factors.append('long_query_complexity')
        
        # 5. Diacritics presence
        if features['has_diacritics']:
            complexity_factors.append('diacritics_present')
            complexity_analysis['processing_recommendations'].append(
                'Normalize diacritics for better matching'
            )
        
        # Calculate overall complexity score
        base_complexity = len(complexity_factors) * 0.2
        ambiguity_contribution = ambiguity['ambiguity_level'] * 0.3
        morphological_contribution = features['morphological_richness'] * 0.2
        
        complexity_analysis['complexity_score'] = min(1.0, 
            base_complexity + ambiguity_contribution + morphological_contribution)
        
        complexity_analysis['complexity_factors'] = complexity_factors
        
        # Add clustering-specific recommendations
        if complexity_analysis['complexity_score'] > 0.7:
            complexity_analysis['clustering_hints'].extend([
                'Use semantic similarity with high weight',
                'Apply context-aware clustering algorithms',
                'Consider ensemble clustering approaches'
            ])
        elif complexity_analysis['complexity_score'] > 0.4:
            complexity_analysis['clustering_hints'].extend([
                'Balance semantic and syntactic features',
                'Use morphological root matching'
            ])
        else:
            complexity_analysis['clustering_hints'].append(
                'Standard clustering approaches should work well'
            )
        
        return complexity_analysis

    def generate_search_expansions(self, query: str, max_expansions: int = 10) -> List[Dict[str, Any]]:
        """
        Generate expanded search terms for better result coverage in Arabic.
        """
        expansions = []
        
        if not self._is_arabic_text(query):
            return expansions
        
        tokens = self.tokenize_arabic(query)
        
        for token in tokens:
            # Morphological expansions
            morphological_variants = self._generate_morphological_variants(token)
            for variant in morphological_variants[:3]:  # Limit per token
                if variant != token:
                    expansions.append({
                        'type': 'morphological',
                        'original': token,
                        'expansion': variant,
                        'confidence': 0.8,
                        'explanation': f'Morphological variant of {token}'
                    })
            
            # Semantic expansions
            semantic_variants = self._generate_semantic_expansions(token)
            for variant in semantic_variants[:2]:  # Limit per token
                expansions.append({
                    'type': 'semantic',
                    'original': token,
                    'expansion': variant,
                    'confidence': 0.6,
                    'explanation': f'Semantic disambiguation of {token}'
                })
            
            # Root-based expansions
            root = self._extract_arabic_root(token)
            if root != token and len(root) >= 3:
                expansions.append({
                    'type': 'root',
                    'original': token,
                    'expansion': root,
                    'confidence': 0.7,
                    'explanation': f'Root form of {token}'
                })
        
        # Sort by confidence and limit
        expansions.sort(key=lambda x: x['confidence'], reverse=True)
        return expansions[:max_expansions]

    def suggest_clustering_parameters(self, query: str) -> Dict[str, Any]:
        """
        Suggest optimal clustering parameters based on Arabic query characteristics.
        """
        complexity = self.analyze_query_complexity(query)
        
        suggestions = {
            'algorithm': 'adaptive',
            'min_cluster_size': 2,
            'similarity_threshold': 0.5,
            'use_morphological_features': True,
            'use_semantic_expansion': False,
            'preprocessing_level': 'standard'
        }
        
        complexity_score = complexity['complexity_score']
        
        # Adjust based on complexity
        if complexity_score > 0.7:
            suggestions.update({
                'algorithm': 'ensemble',
                'min_cluster_size': 3,
                'similarity_threshold': 0.3,
                'use_semantic_expansion': True,
                'preprocessing_level': 'aggressive'
            })
        elif complexity_score > 0.4:
            suggestions.update({
                'algorithm': 'hdbscan',
                'similarity_threshold': 0.4,
                'use_semantic_expansion': True,
                'preprocessing_level': 'moderate'
            })
        
        # Adjust for specific complexity factors
        if 'semantic_ambiguity' in complexity['complexity_factors']:
            suggestions['use_semantic_expansion'] = True
            suggestions['min_cluster_size'] = max(suggestions['min_cluster_size'], 3)
        
        if 'high_morphological_richness' in complexity['complexity_factors']:
            suggestions['use_morphological_features'] = True
            suggestions['preprocessing_level'] = 'aggressive'
        
        if 'single_word_ambiguity' in complexity['complexity_factors']:
            suggestions['algorithm'] = 'gaussian_mixture'  # Better for ambiguous single terms
            suggestions['similarity_threshold'] = 0.2
        
        return suggestions