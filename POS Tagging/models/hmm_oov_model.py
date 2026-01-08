# models/hmm_oov_model.py
import os
import sys
import pickle
import nltk
from nltk.tag import hmm

# Thêm đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

class HMMOOVTagger:
    def __init__(self, model_path=None):
        """
        HMM Tagger với xử lý OOV (Out-Of-Vocabulary)
        """
        self.model = None
        self.model_loaded = False
        self.vocab = set()
        self.OOV_TOKEN = "__OOV__"
        
        if model_path is None:
            # Tìm file model tự động
            model_path = self._find_model_file()
        
        self._load_model(model_path)
    
    def _find_model_file(self):
        """Tìm file HMM OOV model"""
        possible_paths = [
            "saved_models/hmm_oov_model.joblib",  # Joblib format
            "saved_models/hmm_oov_model.pkl",     # Pickle format
            "saved_models/hmm_oov.pkl",           # Tên cũ
            "saved_models/hmm_oov.joblib",
            os.path.join("models", "saved_models", "hmm_oov_model.joblib"),
            "hmm_oov_model.joblib",
            "hmm_oov.pkl"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found HMM-OOV model: {path}")
                return path
        
        print("❌ No HMM-OOV model file found")
        return None
    
    def _load_model(self, model_path):
        """Load HMM OOV model từ file"""
        if model_path and os.path.exists(model_path):
            try:
                # Thử load với joblib trước
                try:
                    import joblib
                    loaded_data = joblib.load(model_path)
                    print("✅ Loaded HMM-OOV model with joblib")
                except:
                    # Thử load với pickle
                    with open(model_path, 'rb') as f:
                        loaded_data = pickle.load(f)
                    print("✅ Loaded HMM-OOV model with pickle")
                
                # Xử lý dữ liệu đã load
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    # Format: (model, vocab)
                    self.model, self.vocab = loaded_data
                elif hasattr(loaded_data, 'tag'):  # Chỉ có model
                    self.model = loaded_data
                    # Cố gắng extract vocab
                    if hasattr(self.model, 'vocab'):
                        self.vocab = self.model.vocab
                    elif hasattr(self.model, '_symbols'):
                        self.vocab = set(self.model._symbols)
                else:
                    self.model = loaded_data
                    self.vocab = set()
                
                self.model_loaded = True
                
                print(f"   Model type: {type(self.model)}")
                print(f"   Vocab size: {len(self.vocab)}" if self.vocab else "   No vocab found")
                print(f"   OOV token: {self.OOV_TOKEN}")
                
            except Exception as e:
                print(f"❌ Failed to load HMM-OOV model: {e}")
                self.model = None
        else:
            print("⚠️ HMM-OOV model file not found")
    
    def _preprocess_tokens(self, tokens):
        """Tiền xử lý tokens và xử lý OOV"""
        processed = []
        
        for token in tokens:
            if not token:
                continue
            
            # Xử lý contractions cơ bản
            lower_token = token.lower()
            contraction_parts = None
            
            if lower_token == "i'm":
                contraction_parts = ["I", "'m"]
            elif lower_token in ["you're", "he's", "she's", "it's", "we're", "they're"]:
                contraction_parts = [token[:-3], "'s"]
            elif lower_token in ["i'll", "you'll", "he'll", "she'll", "it'll", "we'll", "they'll"]:
                contraction_parts = [token[:-4], "'ll"]
            elif lower_token in ["i'd", "you'd", "he'd", "she'd", "it'd", "we'd", "they'd"]:
                contraction_parts = [token[:-3], "'d"]
            elif lower_token in ["i've", "you've", "we've", "they've"]:
                contraction_parts = [token[:-4], "'ve"]
            
            if contraction_parts:
                # Xử lý OOV cho từng phần của contraction
                for part in contraction_parts:
                    if self.vocab and part.lower() not in self.vocab and part not in ["'m", "'s", "'re", "'ll", "'d", "'ve"]:
                        processed.append(self.OOV_TOKEN)
                    else:
                        processed.append(part)
            else:
                # Kiểm tra OOV
                if self.vocab and token.lower() not in self.vocab:
                    processed.append(self.OOV_TOKEN)
                else:
                    processed.append(token)
        
        return processed
    
    def _replace_oov_words(self, tokens, vocab, oov_token="__OOV__"):
        """Thay thế từ OOV bằng OOV token"""
        replaced = []
        for token in tokens:
            if token.lower() in vocab or token in ["'m", "'s", "'re", "'ll", "'d", "'ve"]:
                replaced.append(token)
            else:
                replaced.append(oov_token)
        return replaced
    
    def tag(self, tokens):
        """Gán nhãn POS với xử lý OOV"""
        if not tokens:
            return []
        
        # Nếu không có model, dùng fallback
        if not self.model_loaded or self.model is None:
            print("⚠️ HMM-OOV model not available, using fallback")
            return self._fallback_tag(tokens)
        
        try:
            # Tiền xử lý tokens và xử lý OOV
            processed_tokens = self._preprocess_tokens(tokens)
            
            # Nếu có vocab, thực hiện OOV replacement thêm lần nữa để chắc chắn
            if self.vocab:
                processed_tokens = self._replace_oov_words(processed_tokens, self.vocab, self.OOV_TOKEN)
            
            # HMM tagging
            tagged = self.model.tag(processed_tokens)
            
            # Extract tags
            tags = [tag for _, tag in tagged]
            
            # Map tags trở lại tokens gốc
            final_tags = []
            proc_idx = 0
            
            for token in tokens:
                lower_token = token.lower()
                
                # Kiểm tra nếu token là contraction
                if lower_token in ["i'm", "you're", "he's", "she's", "it's", "we're", "they're",
                                 "i'll", "you'll", "he'll", "she'll", "it'll", "we'll", "they'll",
                                 "i'd", "you'd", "he'd", "she'd", "it'd", "we'd", "they'd",
                                 "i've", "you've", "we've", "they've"]:
                    # Lấy tag của phần đầu contraction
                    if proc_idx < len(tags):
                        final_tags.append(tags[proc_idx])
                    else:
                        final_tags.append("NOUN")
                    proc_idx += 2  # Bỏ qua contraction part
                else:
                    # Token thông thường
                    if proc_idx < len(tags):
                        final_tags.append(tags[proc_idx])
                    else:
                        final_tags.append("NOUN")
                    proc_idx += 1
            
            print(f"✅ HMM-OOV model predicted {len(final_tags)} tags")
            print(f"   OOV replacements: {processed_tokens.count(self.OOV_TOKEN)}")
            
            return final_tags
            
        except Exception as e:
            print(f"❌ HMM-OOV prediction error: {e}")
            return self._fallback_tag(tokens)
    
    def _fallback_tag(self, tokens):
        """Rule-based fallback tagging với OOV handling"""
        tags = []
        for token in tokens:
            if not token:
                tags.append("X")
                continue
                
            lower_token = token.lower()
            
            # OOV detection đơn giản
            if self.vocab and token.lower() not in self.vocab:
                # Từ OOV, dùng heuristic
                if token.endswith('ing'):
                    tags.append("VERB")
                elif token.endswith('ly'):
                    tags.append("ADV")
                elif token[0].isupper():
                    tags.append("PROPN")
                elif any(c.isdigit() for c in token):
                    tags.append("NUM")
                else:
                    tags.append("NOUN")
            else:
                # Từ trong vocab
                if token == "I":
                    tags.append("PRON")
                elif lower_token in ['the', 'a', 'an', 'this', 'that']:
                    tags.append("DET")
                elif lower_token in ['i', 'you', 'he', 'she', 'it', 'we', 'they']:
                    tags.append("PRON")
                elif lower_token in ['is', 'am', 'are', 'was', 'were']:
                    tags.append("VERB")
                elif token.endswith('ing'):
                    tags.append("VERB")
                elif token.endswith('ly'):
                    tags.append("ADV")
                elif token[0].isupper():
                    tags.append("PROPN")
                elif any(c.isdigit() for c in token):
                    tags.append("NUM")
                else:
                    tags.append("NOUN")
        
        return tags


# Test function
def test_hmm_oov_tagger():
    """Test HMM-OOV tagger"""
    print("\n🧪 Testing HMM-OOV Tagger...")
    tagger = HMMOOVTagger()
    
    if tagger.model_loaded:
        print("✅ Model loaded successfully")
        
        # Test với từ OOV và contractions
        test_sentences = [
            "I'm the one who shall grid",  # "grid" có thể là OOV
            "The quixotic fox jumps blithely",  # "quixotic" và "blithely" có thể là OOV
            "Google is a zylophone company",  # "zylophone" là OOV
        ]
        
        for sentence in test_sentences:
            print(f"\n📝 Sentence: {sentence}")
            tokens = sentence.split()
            tags = tagger.tag(tokens)
            
            print("   Results:")
            for token, tag in zip(tokens, tags):
                is_oov = tagger.vocab and token.lower() not in tagger.vocab
                oov_marker = " (OOV)" if is_oov else ""
                print(f"     '{token:15}' -> {tag}{oov_marker}")
    else:
        print("❌ Model not loaded")

if __name__ == "__main__":
    test_hmm_oov_tagger()