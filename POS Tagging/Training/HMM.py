# training/HMM.py
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
print("🤖 TRAINING HMM MODEL FOR POS TAGGING")
print("=" * 60)

def create_training_data():
    """Tạo dữ liệu training cho HMM"""
    print("📝 Creating training data for HMM...")
    
    # Universal Dependencies POS tags
    sentences = [
        # Basic sentences
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
        
        # More sentences for better coverage
        [("This", "DET"), ("is", "AUX"), ("an", "DET"), ("example", "NOUN"), 
         ("sentence", "NOUN"), (".", "PUNCT")],
        
        [("He", "PRON"), ("quickly", "ADV"), ("ran", "VERB"), ("to", "ADP"), 
         ("the", "DET"), ("store", "NOUN"), (".", "PUNCT")],
        
        [("We", "PRON"), ("should", "AUX"), ("consider", "VERB"), ("all", "DET"), 
         ("possibilities", "NOUN"), (".", "PUNCT")],
        
        [("The", "DET"), ("weather", "NOUN"), ("is", "AUX"), ("beautiful", "ADJ"), 
         ("today", "NOUN"), (".", "PUNCT")],
        
        [("I", "PRON"), ("have", "AUX"), ("three", "NUM"), ("apples", "NOUN"), 
         (".", "PUNCT")],
        
        # Contractions (tách riêng)
        [("I", "PRON"), ("'m", "AUX"), ("happy", "ADJ"), (".", "PUNCT")],
        
        [("You", "PRON"), ("'re", "AUX"), ("welcome", "ADJ"), (".", "PUNCT")],
        
        [("He", "PRON"), ("'s", "AUX"), ("coming", "VERB"), (".", "PUNCT")],
        
        [("They", "PRON"), ("'ll", "AUX"), ("arrive", "VERB"), ("soon", "ADV"), 
         (".", "PUNCT")],
        
        [("We", "PRON"), ("'ve", "AUX"), ("finished", "VERB"), (".", "PUNCT")],
    ]
    
    # Statistics
    all_words = []
    all_tags = []
    for sent in sentences:
        for word, tag in sent:
            all_words.append(word)
            all_tags.append(tag)
    
    print(f"✅ Created {len(sentences)} training sentences")
    print(f"   Total tokens: {len(all_words):,}")
    print(f"   Unique words: {len(set(all_words)):,}")
    print(f"   Unique POS tags: {len(set(all_tags))}")
    print(f"   Tag distribution: {Counter(all_tags).most_common(10)}")
    
    return sentences

def train_hmm_model():
    """Train HMM model"""
    print("\n🧠 Training HMM model...")
    
    # Tạo dữ liệu
    data = create_training_data()
    
    # Train HMM model
    try:
        from nltk.tag import hmm
        
        trainer = hmm.HiddenMarkovModelTrainer()
        print("Training HMM model (this may take a moment)...")
        
        # Sử dụng supervised training
        model = trainer.train_supervised(data)
        
        print("✅ HMM model trained successfully!")
        
        # Model information
        print(f"\n📊 Model Information:")
        print(f"   States (POS tags): {len(model._states)}")
        print(f"   Symbols (words): {len(model._symbols)}")
        
        # Test model
        print("\n🧪 Testing trained model...")
        test_sentences = [
            ["The", "fox", "jumps"],
            ["I", "love", "NLP"],
            ["Google", "is", "great"],
        ]
        
        for tokens in test_sentences:
            try:
                tagged = model.tag(tokens)
                print(f"   '{' '.join(tokens)}' -> {[tag for _, tag in tagged]}")
            except:
                print(f"   '{' '.join(tokens)}' -> Error in tagging")
        
        return model, data
        
    except Exception as e:
        print(f"❌ HMM training error: {e}")
        import traceback
        traceback.print_exc()
        return None, data

def save_model(model, vocab=None, filename="hmm_model.joblib"):
    """Lưu model"""
    print(f"\n💾 Saving model to saved_models/{filename}...")
    
    # Tạo thư mục nếu chưa có
    os.makedirs("saved_models", exist_ok=True)
    
    # Chuẩn bị data để lưu
    if vocab is not None:
        save_data = {
            'model': model,
            'vocab': vocab,
            'model_type': 'hmm'
        }
    else:
        save_data = model
    
    # Lưu với joblib (tốt hơn pickle)
    model_path = os.path.join("saved_models", filename)
    joblib.dump(save_data, model_path, compress=3)
    
    # Kiểm tra
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✅ Model saved: {model_path}")
        print(f"   File size: {size:,} bytes ({size/1024:.1f} KB)")
    else:
        print(f"❌ Failed to save model")

def extract_vocab(data):
    """Trích xuất vocabulary từ training data"""
    vocab = set()
    for sent in data:
        for word, _ in sent:
            vocab.add(word.lower())  # Chuyển về lowercase
    
    print(f"📚 Vocabulary extracted: {len(vocab):,} unique words")
    return vocab

if __name__ == "__main__":
    try:
        # Train model
        model, data = train_hmm_model()
        
        if model is not None:
            # Extract vocabulary
            vocab = extract_vocab(data)
            
            # Save model
            save_model(model, vocab, "hmm_model.joblib")
            
            print("\n" + "=" * 60)
            print("🎉 HMM MODEL TRAINING COMPLETED!")
            print("=" * 60)
            
            print("\n📋 Next steps:")
            print("1. Run: streamlit run streamlit_app.py")
            print("2. Select 'HMM' model in sidebar")
            print("3. Test with sample sentences")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()