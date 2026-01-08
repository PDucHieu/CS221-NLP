# training/HMM_OOV.py
import os
import sys
import joblib
import pickle
import nltk
from nltk.tag import hmm
import numpy as np
from collections import Counter, defaultdict

# Thêm đường dẫn để import
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print("=" * 60)
print("🤖 TRAINING HMM WITH OOV HANDLING")
print("=" * 60)

def create_training_data_with_oov():
    """Tạo dữ liệu training với OOV augmentation"""
    print("📝 Creating training data with OOV handling...")
    
    # Base training data (giống HMM thông thường)
    base_sentences = [
        [("The", "DET"), ("quick", "ADJ"), ("brown", "ADJ"), ("fox", "NOUN"), 
         ("jumps", "VERB"), ("over", "ADP"), ("the", "DET"), ("lazy", "ADJ"), 
         ("dog", "NOUN"), (".", "PUNCT")],
        
        [("I", "PRON"), ("love", "VERB"), ("natural", "ADJ"), ("language", "NOUN"), 
         ("processing", "NOUN"), (".", "PUNCT")],
        
        [("Google", "PROPN"), ("is", "AUX"), ("a", "DET"), ("technology", "NOUN"), 
         ("company", "NOUN"), (".", "PUNCT")],
        
        [("She", "PRON"), ("will", "AUX"), ("be", "AUX"), ("arriving", "VERB"), 
         ("at", "ADP"), ("3", "NUM"), ("PM", "NOUN"), ("tomorrow", "NOUN"), 
         (".", "PUNCT")],
        
        [("They", "PRON"), ("are", "AUX"), ("learning", "VERB"), ("machine", "NOUN"), 
         ("learning", "NOUN"), (".", "PUNCT")],
        
        # More sentences
        [("This", "DET"), ("is", "AUX"), ("an", "DET"), ("example", "NOUN"), 
         ("sentence", "NOUN"), (".", "PUNCT")],
        
        [("He", "PRON"), ("quickly", "ADV"), ("ran", "VERB"), ("to", "ADP"), 
         ("the", "DET"), ("store", "NOUN"), (".", "PUNCT")],
        
        [("We", "PRON"), ("should", "AUX"), ("consider", "VERB"), ("all", "DET"), 
         ("possibilities", "NOUN"), (".", "PUNCT")],
    ]
    
    # Thêm OOV examples
    oov_sentences = []
    oov_token = "__OOV__"
    
    # Các pattern OOV thường gặp
    oov_patterns = [
        (oov_token, "NOUN"),    # OOV thường là danh từ
        (oov_token, "VERB"),    # Hoặc động từ
        (oov_token, "ADJ"),     # Hoặc tính từ
        ("__NUM__", "NUM"),     # Số
        ("__UPPER__", "PROPN"), # Từ viết hoa
        ("__TITLE__", "PROPN"), # Từ có chữ cái đầu viết hoa
    ]
    
    # Thêm sentences với OOV
    for pattern, tag in oov_patterns:
        # Tạo câu có chứa OOV
        for base_sent in base_sentences[:5]:  # Lấy 5 câu đầu
            # Thay một từ ngẫu nhiên bằng OOV
            if len(base_sent) > 2:
                import random
                idx = random.randint(0, len(base_sent)-1)
                new_sent = base_sent.copy()
                new_sent[idx] = (pattern, tag)
                oov_sentences.append(new_sent)
    
    # Kết hợp data
    all_sentences = base_sentences + oov_sentences
    
    # Statistics
    all_words = []
    all_tags = []
    for sent in all_sentences:
        for word, tag in sent:
            all_words.append(word)
            all_tags.append(tag)
    
    print(f"✅ Created {len(base_sentences)} base sentences")
    print(f"✅ Added {len(oov_sentences)} OOV-augmented sentences")
    print(f"✅ Total: {len(all_sentences)} training sentences")
    print(f"   Total tokens: {len(all_words):,}")
    print(f"   Unique words: {len(set(all_words)):,}")
    print(f"   Unique POS tags: {len(set(all_tags))}")
    
    # Vocabulary (loại bỏ OOV tokens)
    vocab = set()
    for sent in base_sentences:
        for word, _ in sent:
            vocab.add(word.lower())
    
    print(f"📚 Vocabulary size: {len(vocab):,} words (excluding OOV tokens)")
    
    return all_sentences, vocab

def train_hmm_oov_model():
    """Train HMM model với OOV handling"""
    print("\n🧠 Training HMM model with OOV handling...")
    
    # Tạo dữ liệu
    data, vocab = create_training_data_with_oov()
    
    # Train HMM model
    try:
        from nltk.tag import hmm
        
        trainer = hmm.HiddenMarkovModelTrainer()
        print("Training HMM model with OOV data...")
        
        # Train với data đã augmented
        model = trainer.train_supervised(data)
        
        print("✅ HMM OOV model trained successfully!")
        
        # Model information
        print(f"\n📊 Model Information:")
        print(f"   States (POS tags): {len(model._states)}")
        print(f"   Symbols (words + OOV tokens): {len(model._symbols)}")
        
        # Test với OOV
        print("\n🧪 Testing OOV handling...")
        test_cases = [
            (["The", "quxz", "fox", "jumps"], "quxz is OOV"),
            (["I", "xyzabc", "NLP"], "xyzabc is OOV"),
            (["Google", "123", "company"], "123 is number"),
            (["THE", "COMPANY", "is"], "UPPERCASE words"),
        ]
        
        for tokens, description in test_cases:
            try:
                tagged = model.tag(tokens)
                tags = [tag for _, tag in tagged]
                print(f"   '{' '.join(tokens)}' -> {tags} ({description})")
            except:
                print(f"   '{' '.join(tokens)}' -> Error ({description})")
        
        return model, vocab
        
    except Exception as e:
        print(f"❌ HMM OOV training error: {e}")
        import traceback
        traceback.print_exc()
        return None, vocab

def save_oov_model(model, vocab, filename="hmm_oov_model.joblib"):
    """Lưu model OOV"""
    print(f"\n💾 Saving OOV model to saved_models/{filename}...")
    
    # Tạo thư mục nếu chưa có
    os.makedirs("saved_models", exist_ok=True)
    
    # Chuẩn bị data để lưu
    save_data = {
        'model': model,
        'vocab': vocab,
        'model_type': 'hmm_oov',
        'oov_token': '__OOV__',
        'special_tokens': ['__NUM__', '__UPPER__', '__TITLE__']
    }
    
    # Lưu với joblib
    model_path = os.path.join("saved_models", filename)
    joblib.dump(save_data, model_path, compress=3)
    
    # Kiểm tra
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✅ Model saved: {model_path}")
        print(f"   File size: {size:,} bytes ({size/1024:.1f} KB)")
        
        # Test load
        try:
            loaded = joblib.load(model_path)
            print(f"   Can load: {loaded is not None}")
            print(f"   Has vocab: 'vocab' in loaded")
        except:
            print("   Load test failed")
    else:
        print(f"❌ Failed to save model")

if __name__ == "__main__":
    try:
        # Train model
        model, vocab = train_hmm_oov_model()
        
        if model is not None:
            # Save model
            save_oov_model(model, vocab, "hmm_oov_model.joblib")
            
            print("\n" + "=" * 60)
            print("🎉 HMM OOV MODEL TRAINING COMPLETED!")
            print("=" * 60)
            
            print("\n📋 Next steps:")
            print("1. Run: streamlit run streamlit_app.py")
            print("2. Select 'HMM with OOV' model in sidebar")
            print("3. Test with sentences containing unknown words")
            print("4. The model should handle OOV words better")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()