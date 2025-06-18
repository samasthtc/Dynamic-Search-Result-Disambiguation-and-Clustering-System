"""
Arabic Text Processor - Handles Arabic text preprocessing as specified in the research paper
Uses pyarabic for normalization and preprocessing
"""

import re
import logging
from typing import List, Dict, Any, Optional

import pyarabic.araby as araby
import pyarabic.trans as trans
from pyarabic.arabrepr import ArabicRepr

logger = logging.getLogger(__name__)

class ArabicProcessor:
    """
    Arabic text processing pipeline as specified in the research paper:
    - Remove diacritics
    - Standardize Alif/Hamza forms
    - Delete stopwords
    - Optional light stemming (reduces vocabulary by 23%)
    """
    
    def __init__(self):
        """Initialize Arabic processor"""
        self.arabic_repr = ArabicRepr()
        
        # Arabic stopwords (common words to remove)
        self.stopwords = {
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
            'التي', 'الذي', 'التي', 'اللذان', 'اللتان', 'اللذين', 'اللتين', 'اللواتي',
            'هو', 'هي', 'هم', 'هن', 'أنت', 'أنتم', 'أنتن', 'أنا', 'نحن',
            'كان', 'كانت', 'كانوا', 'كن', 'يكون', 'تكون', 'يكونوا', 'يكن',
            'له', 'لها', 'لهم', 'لهن', 'لك', 'لكم', 'لكن', 'لي', 'لنا',
            'به', 'بها', 'بهم', 'بهن', 'بك', 'بكم', 'بكن', 'بي', 'بنا',
            'قد', 'لقد', 'كل', 'جميع', 'بعض', 'غير', 'سوى', 'إلا', 'لكن',
            'أو', 'أم', 'لا', 'ما', 'لم', 'لن', 'إن', 'أن', 'كي', 'حتى'
        }
        
        logger.info("Arabic processor initialized")
    
    def remove_diacritics(self, text: str) -> str:
        """
        Remove Arabic diacritics (harakat) from text
        
        Args:
            text: Input Arabic text with diacritics
            
        Returns:
            Text without diacritics
        """
        try:
            return araby.strip_diacritics(text)
        except Exception as e:
            logger.warning(f"Error removing diacritics: {e}")
            return text
    
    def normalize_alif_hamza(self, text: str) -> str:
        """
        Standardize Alif and Hamza forms
        
        Args:
            text: Input Arabic text
            
        Returns:
            Text with standardized Alif/Hamza
        """
        try:
            # Normalize different Alif forms to standard Alif
            text = araby.normalize_alef(text)
            
            # Normalize Hamza forms
            text = araby.normalize_hamza(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error normalizing Alif/Hamza: {e}")
            return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove Arabic stopwords
        
        Args:
            text: Input Arabic text
            
        Returns:
            Text without stopwords
        """
        try:
            words = text.split()
            filtered_words = [word for word in words if word not in self.stopwords]
            return ' '.join(filtered_words)
        except Exception as e:
            logger.warning(f"Error removing stopwords: {e}")
            return text
    
    def tokenize_arabic(self, text: str) -> List[str]:
        """
        Tokenize Arabic text
        
        Args:
            text: Input Arabic text
            
        Returns:
            List of tokens
        """
        try:
            # Simple Arabic tokenization
            tokens = araby.tokenize(text)
            return [token for token in tokens if araby.is_arabicword(token)]
        except Exception as e:
            logger.warning(f"Error tokenizing: {e}")
            return text.split()
    
    def light_stemming(self, text: str) -> str:
        """
        Apply light stemming to reduce vocabulary size by ~23% as mentioned in paper
        
        Args:
            text: Input Arabic text
            
        Returns:
            Text with light stemming applied
        """
        try:
            words = text.split()
            stemmed_words = []
            
            for word in words:
                if araby.is_arabicword(word):
                    # Remove common prefixes
                    stemmed = araby.strip_tatweel(word)
                    stemmed = self._remove_prefixes(stemmed)
                    stemmed = self._remove_suffixes(stemmed)
                    stemmed_words.append(stemmed)
                else:
                    stemmed_words.append(word)
            
            return ' '.join(stemmed_words)
            
        except Exception as e:
            logger.warning(f"Error in light stemming: {e}")
            return text
    
    def _remove_prefixes(self, word: str) -> str:
        """Remove common Arabic prefixes"""
        prefixes = ['ال', 'و', 'ف', 'ب', 'ك', 'ل', 'لل']
        
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                return word[len(prefix):]
        
        return word
    
    def _remove_suffixes(self, word: str) -> str:
        """Remove common Arabic suffixes"""
        suffixes = ['ها', 'ان', 'ات', 'ون', 'ين', 'تم', 'كم', 'هن', 'نا', 'ني', 'وا', 'تن', 'ة', 'ه']
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        
        return word
    
    def preprocess_text(self, text: str, apply_stemming: bool = True) -> str:
        """
        Complete Arabic preprocessing pipeline as specified in the paper:
        1. Remove diacritics
        2. Standardize Alif/Hamza
        3. Remove stopwords
        4. Tokenize
        5. Optional light stemming
        
        Args:
            text: Input Arabic text
            apply_stemming: Whether to apply light stemming
            
        Returns:
            Preprocessed text
        """
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Step 1: Remove diacritics
            processed_text = self.remove_diacritics(text)
            
            # Step 2: Standardize Alif/Hamza forms
            processed_text = self.normalize_alif_hamza(processed_text)
            
            # Step 3: Clean up extra whitespace and punctuation
            processed_text = re.sub(r'[^\w\s]', ' ', processed_text)
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            
            # Step 4: Remove stopwords
            processed_text = self.remove_stopwords(processed_text)
            
            # Step 5: Optional light stemming
            if apply_stemming:
                processed_text = self.light_stemming(processed_text)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error in Arabic preprocessing: {e}")
            return text  # Return original text if preprocessing fails
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess Arabic search query
        
        Args:
            query: Input search query in Arabic
            
        Returns:
            Preprocessed query
        """
        return self.preprocess_text(query, apply_stemming=True)
    
    def is_arabic_text(self, text: str) -> bool:
        """
        Check if text contains Arabic characters
        
        Args:
            text: Input text
            
        Returns:
            True if text contains Arabic characters
        """
        try:
            return bool(re.search(r'[\u0600-\u06FF]', text))
        except Exception as e:
            logger.warning(f"Error checking Arabic text: {e}")
            return False
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about Arabic text processing
        
        Args:
            text: Input Arabic text
            
        Returns:
            Dictionary with text statistics
        """
        try:
            original_words = len(text.split())
            preprocessed = self.preprocess_text(text, apply_stemming=False)
            preprocessed_words = len(preprocessed.split())
            
            stemmed = self.preprocess_text(text, apply_stemming=True)
            stemmed_words = len(stemmed.split())
            
            # Calculate vocabulary reduction
            vocab_reduction = ((original_words - stemmed_words) / original_words * 100) if original_words > 0 else 0
            
            return {
                'original_word_count': original_words,
                'preprocessed_word_count': preprocessed_words,
                'stemmed_word_count': stemmed_words,
                'vocabulary_reduction_pct': vocab_reduction,
                'contains_arabic': self.is_arabic_text(text),
                'diacritics_removed': len(text) - len(self.remove_diacritics(text))
            }
            
        except Exception as e:
            logger.error(f"Error calculating text statistics: {e}")
            return {
                'original_word_count': 0,
                'preprocessed_word_count': 0,
                'stemmed_word_count': 0,
                'vocabulary_reduction_pct': 0,
                'contains_arabic': False,
                'diacritics_removed': 0
            }
    
    def handle_disambiguation_terms(self, text: str) -> str:
        """
        Handle Arabic disambiguation terms and homographs
        
        Args:
            text: Input text with potential ambiguous terms
            
        Returns:
            Text with disambiguation handling
        """
        try:
            # Common Arabic homographs and their contexts
            disambiguation_map = {
                'عين': ['eye', 'spring', 'letter_ain'],  # عين can mean eye, water spring, or letter
                'ورد': ['rose', 'arrived', 'mentioned'],  # ورد can mean rose or past tense of arrive
                'بنك': ['bank_financial', 'bank_river'],  # بنك - financial bank or river bank
                'تفاحة': ['apple_fruit'],  # Apple fruit (less ambiguous in Arabic)
                'جاكسون': ['jackson_person', 'jackson_place']  # Jackson transliteration
            }
            
            # This is a simplified approach - in practice, you'd use more sophisticated
            # context analysis and possibly machine learning models
            processed_text = text
            
            for term, contexts in disambiguation_map.items():
                if term in processed_text:
                    # For now, just mark the term for potential clustering
                    # In a full implementation, you'd analyze surrounding context
                    logger.debug(f"Found potentially ambiguous term: {term}")
            
            return processed_text
            
        except Exception as e:
            logger.warning(f"Error in disambiguation handling: {e}")
            return text