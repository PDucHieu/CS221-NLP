# models/hmm_model.py
import nltk
from nltk.tag import hmm
import joblib
import os
import sys

# Thêm đường dẫn để import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from preprocessing.text_utils import replace_oov_words

class HMMTagger:
    def __init__(self, model_path=None):
        """
        Khởi tạo HMM Tagger (không xử lý OOV)
        """
        if model_path is None:
            possible_paths = [
                "saved_models/hmm_model.joblib",
                "models/saved_models/hmm_model.joblib"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.model = joblib.load(path)
                    self.vocab = getattr(self.model, 'vocab', set())
                    print(f"✅ Loaded HMM model from: {path}")
                    break
            else:
                raise FileNotFoundError("HMM model file not found")
        else:
            self.model = joblib.load(model_path)
            self.vocab = getattr(self.model, 'vocab', set())
    
    def build_vocab(self, sentences):
        """Xây dựng từ điển từ tập sentences"""
        vocab = set()
        for sent in sentences:
            for word, _ in sent:
                vocab.add(word)
        return vocab
    
    def tag(self, tokens):
        """Gán nhãn POS cho tokens (không xử lý OOV)"""
        try:
            tagged = self.model.tag(tokens)
            return [tag for _, tag in tagged]
        except Exception as e:
            # Fallback: trả về NOUN cho tất cả
            print(f"HMM tagging error: {e}")
            return ['NOUN'] * len(tokens)
    
    def tag_sentence(self, sentence):
        """Gán nhãn cho một câu"""
        tokens = sentence.split()
        return self.tag(tokens)