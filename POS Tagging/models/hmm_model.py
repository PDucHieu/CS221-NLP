# models/hmm_model.py
import os
import sys
import pickle
import nltk
from nltk.tag import hmm

# Thêm đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

class HMMTagger:
    def __init__(self, model_path=None):
        """
        HMM Tagger (không xử lý OOV)
        """
        self.model = None
        self.model_loaded = False
        self.vocab = set()
        
        if model_path is None:
            # Tìm file model tự động
            model_path = self._find_model_file()
        
        self._load_model(model_path)
    
    def _find_model_file(self):
        """Tìm file HMM model"""
        possible_paths = [
            "saved_models/hmm_model.joblib",  # Joblib format
            "saved_models/hmm_model.pkl",     # Pickle format  
            "saved_models/hmm.pkl",           # Tên cũ
            "saved_models/hmm.joblib",
            os.path.join("models", "saved_models", "hmm_model.joblib"),
            "hmm_model.joblib",
            "hmm.pkl"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found HMM model: {path}")
                return path
        
        print("❌ No HMM model file found")
        return None
    
    def _load_model(self, model_path):
        """Load HMM model từ file"""
        if model_path and os.path.exists(model_path):
            try:
                # Thử load với joblib trước
                try:
                    import joblib
                    self.model = joblib.load(model_path)
                    print("✅ Loaded HMM model with joblib")
                except:
                    # Thử load với pickle
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    print("✅ Loaded HMM model with pickle")
                
                self.model_loaded = True
                
                # Extract vocab từ model nếu có
                if hasattr(self.model, 'vocab'):
                    self.vocab = self.model.vocab
                elif hasattr(self.model, '_symbols'):
                    self.vocab = set(self.model._symbols)
                
                print(f"   Model type: {type(self.model)}")
                print(f"   Vocab size: {len(self.vocab)}" if self.vocab else "   No vocab found")
                
            except Exception as e:
                print(f"❌ Failed to load HMM model: {e}")
                self.model = None
        else:
            print("⚠️ HMM model file not found")
    
    def _preprocess_tokens(self, tokens):
        """Tiền xử lý tokens cho HMM"""
        processed = []
        for token in tokens:
            if not token:
                continue
            
            # Xử lý contractions cơ bản
            lower_token = token.lower()
            if lower_token == "i'm":
                processed.extend(["I", "'m"])
            elif lower_token in ["you're", "he's", "she's", "it's", "we're", "they're"]:
                processed.extend([token[:-3], "'s"])
            elif lower_token in ["i'll", "you'll", "he'll", "she'll", "it'll", "we'll", "they'll"]:
                processed.extend([token[:-4], "'ll"])
            elif lower_token in ["i'd", "you'd", "he'd", "she'd", "it'd", "we'd", "they'd"]:
                processed.extend([token[:-3], "'d"])
            elif lower_token in ["i've", "you've", "we've", "they've"]:
                processed.extend([token[:-4], "'ve"])
            else:
                processed.append(token)
        
        return processed
    
    def tag(self, tokens):
        """Gán nhãn POS cho tokens"""
        if not tokens:
            return []
        
        # Nếu không có model, dùng fallback
        if not self.model_loaded or self.model is None:
            print("⚠️ HMM model not available, using fallback")
            return self._fallback_tag(tokens)
        
        try:
            # Tiền xử lý tokens
            processed_tokens = self._preprocess_tokens(tokens)
            
            # HMM tagging
            tagged = self.model.tag(processed_tokens)
            
            # Extract tags
            tags = [tag for _, tag in tagged]
            
            # Nếu số tags khác số tokens (do xử lý contractions)
            if len(tags) != len(tokens):
                # Map back
                if len(processed_tokens) > len(tokens):
                    # Đã tách contractions
                    original_tags = []
                    proc_idx = 0
                    for token in tokens:
                        lower_token = token.lower()
                        if lower_token in ["i'm", "you're", "he's", "she's", "it's", "we're", "they're",
                                         "i'll", "you'll", "he'll", "she'll", "it'll", "we'll", "they'll",
                                         "i'd", "you'd", "he'd", "she'd", "it'd", "we'd", "they'd",
                                         "i've", "you've", "we've", "they've"]:
                            # Lấy tag của từ đầu tiên trong contraction
                            original_tags.append(tags[proc_idx])
                            proc_idx += 2  # Bỏ qua contraction part
                        else:
                            original_tags.append(tags[proc_idx])
                            proc_idx += 1
                    tags = original_tags
            
            print(f"✅ HMM model predicted {len(tags)} tags")
            return tags
            
        except Exception as e:
            print(f"❌ HMM prediction error: {e}")
            return self._fallback_tag(tokens)
    
    def _fallback_tag(self, tokens):
        """Rule-based fallback tagging"""
        tags = []
        for token in tokens:
            if not token:
                tags.append("X")
                continue
                
            lower_token = token.lower()
            
            if token == "I":
                tags.append("PRON")
            elif lower_token in ['the', 'a', 'an', 'this', 'that']:
                tags.append("DET")
            elif lower_token in ['i', 'you', 'he', 'she', 'it', 'we', 'they']:
                tags.append("PRON")
            elif lower_token in ['is', 'am', 'are', 'was', 'were', 'be', 'been']:
                tags.append("VERB")
            elif token.endswith('ing'):
                tags.append("VERB")
            elif token.endswith('ly'):
                tags.append("ADV")
            elif token[0].isupper() and len(token) > 1:
                tags.append("PROPN")
            elif any(c.isdigit() for c in token):
                tags.append("NUM")
            else:
                tags.append("NOUN")
        
        return tags


# Test function
def test_hmm_tagger():
    """Test HMM tagger"""
    print("\n🧪 Testing HMM Tagger...")
    tagger = HMMTagger()
    
    if tagger.model_loaded:
        print("✅ Model loaded successfully")
        
        test_sentences = [
            "I love natural language processing",
            "The quick brown fox jumps",
            "Google is a company",
        ]
        
        for sentence in test_sentences:
            print(f"\n📝 Sentence: {sentence}")
            tokens = sentence.split()
            tags = tagger.tag(tokens)
            
            for token, tag in zip(tokens, tags):
                print(f"   {token:15} -> {tag}")
    else:
        print("❌ Model not loaded")

if __name__ == "__main__":
    test_hmm_tagger()