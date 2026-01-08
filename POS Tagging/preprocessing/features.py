# preprocessing/features.py
import re
import string
import nltk
from nltk.corpus import stopwords

# Tải stopwords
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def get_word_shape(word):
    """Biến đổi từ thành dạng shape pattern"""
    if not word:
        return 'EMPTY'
    shape = re.sub(r'[A-Z]', 'X', word)
    shape = re.sub(r'[a-z]', 'x', shape)
    shape = re.sub(r'[0-9]', 'd', shape)
    return shape

def get_char_ngrams(word, n=2):
    """Trích xuất character n-grams"""
    if len(word) < n:
        return [word]
    return [word[i:i+n] for i in range(len(word)-n+1)]

def word2features(sent, i):
    """Trích xuất features cho một từ trong câu"""
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.len': len(word),
        'word[-3:]': word[-3:] if len(word) >= 3 else word,
        'word[-2:]': word[-2:] if len(word) >= 2 else word,
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.shape': get_word_shape(word),
        'word.is_stopword': word.lower() in stop_words,
    }
    
    # Context features
    if i > 0:
        prev_word = sent[i-1][0]
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.istitle()': prev_word.istitle(),
        })
    else:
        features['BOS'] = True
    
    if i < len(sent) - 1:
        next_word = sent[i+1][0]
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.istitle()': next_word.istitle(),
        })
    else:
        features['EOS'] = True
    
    return features

def extract_features(sent):
    """Trích xuất features cho toàn bộ câu"""
    return [word2features(sent, i) for i in range(len(sent))]

def get_labels(sent):
    """Lấy labels từ câu"""
    return [pos for _, pos in sent]