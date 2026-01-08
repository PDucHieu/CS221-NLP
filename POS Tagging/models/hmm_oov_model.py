# models/hmm_oov_model.py
import nltk
from nltk.tag import hmm
import joblib
import os
import sys

# Thêm đường dẫn để import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from preprocessing.text_utils import replace_oov_words, augment_training_with_oov

class HMMOOVTagger:
    def __init__(self, model_path=None, vocab_path=None):
        """
        Khởi tạo HMM Tagger với xử lý OOV
        """
        if model_path is None:
            possible_paths = [
                "saved_models/hmm_oov_model.joblib",
                "models/saved_models/hmm_oov_model.joblib"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.model = joblib.load(path)
                    self.vocab = getattr(self.model, 'vocab', set())
                    print(f"✅ Loaded HMM OOV model from: {path}")
                    break
            else:
                raise FileNotFoundError("HMM OOV model file not found")
        else:
            self.model = joblib.load(model_path)
            self.vocab = getattr(self.model, 'vocab', set())
    
    def tag(self, tokens):
        """Gán nhãn POS với xử lý OOV"""
        # Thay thế OOV tokens
        if not hasattr(self, 'vocab') or not self.vocab:
            # Nếu không có vocab, xử lý như HMM thông thường
            try:
                tagged = self.model.tag(tokens)
                return [tag for _, tag in tagged]
            except:
                return ['NOUN'] * len(tokens)
        
        # Có vocab, thực hiện OOV replacement
        replaced_tokens = replace_oov_words(tokens, self.vocab, oov_token="__OOV__")
        
        try:
            tagged = self.model.tag(replaced_tokens)
            return [tag for _, tag in tagged]
        except Exception as e:
            print(f"HMM OOV tagging error: {e}")
            return ['NOUN'] * len(tokens)
    
    def tag_sentence(self, sentence):
        """Gán nhãn cho một câu với xử lý OOV"""
        tokens = sentence.split()
        return self.tag(tokens)