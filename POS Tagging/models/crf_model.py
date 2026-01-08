# models/crf_model.py
import joblib
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)

class CRFTagger:
    def __init__(self, model_path=None):
        """
        CRF Tagger - Phiên bản fix cho model đã train
        """
        self.model = None
        self.model_loaded = False
        
        if model_path is None:
            # Tìm file model với cả 2 tên có thể
            model_path = self._find_model_file()
        
        self._load_model(model_path)
    
    def _find_model_file(self):
        """Tìm file model CRF"""
        possible_names = [
            "saved_models/crf_model.joblib",  # Tên mới
            "saved_models/crf.joblib",        # Tên từ training của bạn
            "models/saved_models/crf_model.joblib",
            "crf_model.joblib",
            "crf.joblib"
        ]
        
        for path in possible_names:
            if os.path.exists(path):
                print(f"✅ Found model file: {path}")
                return path
        
        print("❌ No CRF model file found")
        return None
    
    def _load_model(self, model_path):
        """Load CRF model từ file"""
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.model_loaded = True
                print(f"✅ CRF model loaded from: {model_path}")
                print(f"   Model type: {type(self.model)}")
                
                # Kiểm tra model
                if hasattr(self.model, 'predict'):
                    print("✅ Model has predict method")
                else:
                    print("⚠️ Model missing predict method")
                    
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                self.model = None
        else:
            print("⚠️ Model file not found or path invalid")
    
    def _extract_features(self, sent):
        """
        Trích xuất features ĐÚNG NHƯ KHI TRAINING
        Phải khớp với hàm extract_features trong preprocessing.features
        """
        # sent là list of (word, pos) nhưng pos để trống khi inference
        features = []
        for i, (word, _) in enumerate(sent):
            feat = {
                'bias': 1.0,
                'word.lower()': word.lower(),
                'word[-3:]': word[-3:] if len(word) >= 3 else word,
                'word[-2:]': word[-2:] if len(word) >= 2 else word,
                'word.isupper()': word.isupper(),
                'word.istitle()': word.istitle(),
                'word.isdigit()': word.isdigit(),
                'word.is_stopword': word.lower() in self._get_stopwords(),
            }
            
            # Word shape
            shape = self._get_word_shape(word)
            feat['word.shape'] = shape
            
            # Prefix và suffix
            if len(word) >= 1:
                feat['prefix-1'] = word[0]
                feat['suffix-1'] = word[-1]
            if len(word) >= 2:
                feat['prefix-2'] = word[:2]
                feat['suffix-2'] = word[-2:]
            if len(word) >= 3:
                feat['prefix-3'] = word[:3]
                feat['suffix-3'] = word[-3:]
            
            # Context features - KHỚP VỚI TRAINING
            if i > 0:
                prev_word = sent[i-1][0]
                feat.update({
                    '-1:word.lower()': prev_word.lower(),
                    '-1:word.istitle()': prev_word.istitle(),
                    '-1:word.isupper()': prev_word.isupper(),
                })
            else:
                feat['BOS'] = True
            
            if i < len(sent) - 1:
                next_word = sent[i+1][0]
                feat.update({
                    '+1:word.lower()': next_word.lower(),
                    '+1:word.istitle()': next_word.istitle(),
                    '+1:word.isupper()': next_word.isupper(),
                })
            else:
                feat['EOS'] = True
            
            features.append(feat)
        
        return features
    
    def _get_stopwords(self):
        """Get English stopwords"""
        try:
            import nltk
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))
        except:
            # Fallback stopwords
            return {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def _get_word_shape(self, word):
        """Get word shape pattern"""
        import re
        if not word:
            return 'EMPTY'
        shape = re.sub(r'[A-Z]', 'X', word)
        shape = re.sub(r'[a-z]', 'x', shape)
        shape = re.sub(r'[0-9]', 'd', shape)
        return shape
    
    def tag(self, tokens):
        """Tag tokens với CRF model đã train"""
        if not tokens:
            return []
        
        # Nếu không có model, dùng fallback
        if not self.model_loaded or self.model is None:
            print("⚠️ CRF model not available, using fallback")
            return self._rule_based_tag(tokens)
        
        try:
            # Tạo sentence format như training: [(word, "")]
            sent = [(token, "") for token in tokens]
            
            # Trích xuất features
            X = self._extract_features(sent)
            
            # Dự đoán - format phải khớp với training
            # Training dùng: crf.fit(X, y) với X là list of sequences
            # Vậy predict cần: crf.predict([X])
            tags = self.model.predict([X])[0]
            
            print(f"✅ CRF model predicted {len(tags)} tags")
            return list(tags)
            
        except Exception as e:
            print(f"❌ CRF prediction error: {e}")
            import traceback
            traceback.print_exc()
            return self._rule_based_tag(tokens)
    
    def _rule_based_tag(self, tokens):
        """Fallback tagging"""
        tags = []
        for token in tokens:
            if not token:
                tags.append("X")
                continue
                
            lower_token = token.lower()
            
            # POS rules
            if token == "I":
                tags.append("PRON")
            elif token in [".", ",", "!", "?", ";", ":", "'", "\""]:
                tags.append("PUNCT")
            elif lower_token in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
                tags.append("DET")
            elif lower_token in ['i', 'you', 'he', 'she', 'it', 'we', 'they']:
                tags.append("PRON")
            elif lower_token in ['is', 'am', 'are', 'was', 'were', 'be', 'been']:
                tags.append("AUX")  # Universal Dependencies dùng AUX
            elif token.endswith('ing'):
                tags.append("VERB")
            elif token.endswith('ly'):
                tags.append("ADV")
            elif token.endswith(('able', 'ible', 'ful', 'ous', 'ive', 'al')):
                tags.append("ADJ")
            elif token[0].isupper() and len(token) > 1:
                tags.append("PROPN")
            elif any(c.isdigit() for c in token):
                tags.append("NUM")
            else:
                tags.append("NOUN")
        
        return tags


# Test
def test():
    print("🧪 Testing CRF Tagger with trained model...")
    tagger = CRFTagger()
    
    if tagger.model_loaded:
        print("✅ Model loaded successfully")
        
        # Test với câu đơn giản
        test_sentence = "I love natural language processing"
        tokens = test_sentence.split()
        
        print(f"\nSentence: {test_sentence}")
        tags = tagger.tag(tokens)
        
        print("\nResults:")
        for token, tag in zip(tokens, tags):
            print(f"  {token:15} -> {tag}")
    else:
        print("❌ Model not loaded")

if __name__ == "__main__":
    test()