# models/model_trainer.py
import os
import sys
import joblib
import pickle
import numpy as np
from collections import Counter

# Thêm đường dẫn để import
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print("🤖 MODEL TRAINER - Training all POS tagging models")

def create_synthetic_data():
    """Tạo dữ liệu synthetic nếu không có data thật"""
    print("📝 Creating synthetic training data...")
    
    # Universal Dependencies format sentences
    synthetic_data = [
        # Basic sentences
        [
            ("The", "DET"),
            ("quick", "ADJ"),
            ("brown", "ADJ"),
            ("fox", "NOUN"),
            ("jumps", "VERB"),
            ("over", "ADP"),
            ("the", "DET"),
            ("lazy", "ADJ"),
            ("dog", "NOUN"),
            (".", "PUNCT")
        ],
        [
            ("I", "PRON"),
            ("love", "VERB"),
            ("natural", "ADJ"),
            ("language", "NOUN"),
            ("processing", "NOUN"),
            (".", "PUNCT")
        ],
        [
            ("She", "PRON"),
            ("will", "AUX"),
            ("be", "AUX"),
            ("arriving", "VERB"),
            ("at", "ADP"),
            ("3", "NUM"),
            ("PM", "NOUN"),
            ("tomorrow", "NOUN"),
            (".", "PUNCT")
        ],
        [
            ("Google", "PROPN"),
            ("is", "AUX"),
            ("a", "DET"),
            ("technology", "NOUN"),
            ("company", "NOUN"),
            (".", "PUNCT")
        ],
        [
            ("They", "PRON"),
            ("are", "AUX"),
            ("learning", "VERB"),
            ("machine", "NOUN"),
            ("learning", "NOUN"),
            (".", "PUNCT")
        ],
        # More varied sentences
        [
            ("My", "DET"),
            ("computer", "NOUN"),
            ("works", "VERB"),
            ("very", "ADV"),
            ("fast", "ADJ"),
            (".", "PUNCT")
        ],
        [
            ("John", "PROPN"),
            ("and", "CONJ"),
            ("Mary", "PROPN"),
            ("went", "VERB"),
            ("to", "ADP"),
            ("the", "DET"),
            ("park", "NOUN"),
            (".", "PUNCT")
        ],
        [
            ("This", "DET"),
            ("is", "AUX"),
            ("an", "DET"),
            ("excellent", "ADJ"),
            ("book", "NOUN"),
            (".", "PUNCT")
        ],
        [
            ("We", "PRON"),
            ("should", "AUX"),
            ("consider", "VERB"),
            ("all", "DET"),
            ("options", "NOUN"),
            ("carefully", "ADV"),
            (".", "PUNCT")
        ],
        [
            ("The", "DET"),
            ("weather", "NOUN"),
            ("is", "AUX"),
            ("beautiful", "ADJ"),
            ("today", "NOUN"),
            (".", "PUNCT")
        ],
        # Contractions
        [
            ("I", "PRON"),
            ("'m", "AUX"),
            ("going", "VERB"),
            ("to", "PART"),
            ("the", "DET"),
            ("store", "NOUN"),
            (".", "PUNCT")
        ],
        [
            ("He", "PRON"),
            ("'s", "AUX"),
            ("a", "DET"),
            ("great", "ADJ"),
            ("teacher", "NOUN"),
            (".", "PUNCT")
        ],
        [
            ("We", "PRON"),
            ("'ll", "AUX"),
            ("see", "VERB"),
            ("you", "PRON"),
            ("later", "ADV"),
            (".", "PUNCT")
        ],
        # Questions
        [
            ("What", "PRON"),
            ("is", "AUX"),
            ("your", "DET"),
            ("name", "NOUN"),
            ("?", "PUNCT")
        ],
        [
            ("How", "ADV"),
            ("are", "AUX"),
            ("you", "PRON"),
            ("?", "PUNCT")
        ],
        # Numbers and symbols
        [
            ("I", "PRON"),
            ("have", "VERB"),
            ("2", "NUM"),
            ("dogs", "NOUN"),
            ("and", "CONJ"),
            ("1", "NUM"),
            ("cat", "NOUN"),
            (".", "PUNCT")
        ],
        [
            ("The", "DET"),
            ("price", "NOUN"),
            ("is", "AUX"),
            ("$", "SYM"),
            ("100", "NUM"),
            (".", "PUNCT")
        ],
        # Adverbs and adjectives
        [
            ("She", "PRON"),
            ("speaks", "VERB"),
            ("English", "PROPN"),
            ("fluently", "ADV"),
            (".", "PUNCT")
        ],
        [
            ("It", "PRON"),
            ("was", "AUX"),
            ("a", "DET"),
            ("really", "ADV"),
            ("interesting", "ADJ"),
            ("movie", "NOUN"),
            (".", "PUNCT")
        ],
        [
            ("He", "PRON"),
            ("ran", "VERB"),
            ("quickly", "ADV"),
            ("to", "ADP"),
            ("catch", "VERB"),
            ("the", "DET"),
            ("bus", "NOUN"),
            (".", "PUNCT")
        ]
    ]
    
    # Statistics
    all_words = []
    all_tags = []
    for sent in synthetic_data:
        for word, tag in sent:
            all_words.append(word)
            all_tags.append(tag)
    
    print(f"✅ Created {len(synthetic_data)} sentences")
    print(f"   Total tokens: {len(all_words)}")
    print(f"   Unique words: {len(set(all_words))}")
    print(f"   Unique POS tags: {len(set(all_tags))}")
    print(f"   Tag distribution: {Counter(all_tags).most_common(10)}")
    
    return synthetic_data

def extract_features_simple(sent):
    """Trích xuất features đơn giản cho CRF"""
    features = []
    for i, (word, pos) in enumerate(sent):
        feat = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:] if len(word) >= 3 else word,
            'word[-2:]': word[-2:] if len(word) >= 2 else word,
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.length': len(word),
        }
        
        # Prefix và suffix
        if len(word) >= 1:
            feat['prefix1'] = word[0].lower()
            feat['suffix1'] = word[-1].lower()
        if len(word) >= 2:
            feat['prefix2'] = word[:2].lower()
            feat['suffix2'] = word[-2:].lower()
        
        # Context
        if i > 0:
            prev_word, prev_pos = sent[i-1]
            feat.update({
                '-1:word.lower()': prev_word.lower(),
                '-1:word.istitle()': prev_word.istitle(),
            })
        else:
            feat['BOS'] = True
        
        if i < len(sent) - 1:
            next_word, next_pos = sent[i+1]
            feat.update({
                '+1:word.lower()': next_word.lower(),
                '+1:word.istitle()': next_word.istitle(),
            })
        else:
            feat['EOS'] = True
        
        features.append(feat)
    
    return features

def get_labels(sent):
    """Lấy labels từ sentence"""
    return [pos for _, pos in sent]

def train_crf_model(train_sents=None, dev_sents=None, model_path="saved_models/crf_model.joblib"):
    """Huấn luyện CRF model"""
    print("\n" + "="*60)
    print("🤖 TRAINING CRF MODEL")
    print("="*60)
    
    # Sử dụng dữ liệu có sẵn hoặc tạo synthetic
    if train_sents is None or dev_sents is None:
        print("⚠️ No training data provided, using synthetic data")
        data = create_synthetic_data()
        # Split thành train và dev
        split_idx = int(len(data) * 0.8)
        train_sents = data[:split_idx]
        dev_sents = data[split_idx:]
    else:
        print(f"✅ Using provided data: {len(train_sents)} train, {len(dev_sents)} dev")
    
    # Kết hợp train và dev
    all_sents = train_sents + dev_sents
    
    print(f"📊 Total sentences for training: {len(all_sents)}")
    
    # Trích xuất features
    print("🔧 Extracting features...")
    X_all = [extract_features_simple(s) for s in all_sents]
    y_all = [get_labels(s) for s in all_sents]
    
    total_tokens = sum(len(x) for x in X_all)
    print(f"✅ Extracted features for {total_tokens} tokens")
    
    # Train CRF model
    print("🧠 Training CRF model...")
    
    try:
        from sklearn_crfsuite import CRF
        
        crf = CRF(
            algorithm='lbfgs',
            c1=0.1,  # L1 regularization
            c2=0.1,  # L2 regularization
            max_iterations=100,
            all_possible_transitions=True,
            verbose=True
        )
        
        crf.fit(X_all, y_all)
        
        print("✅ CRF model trained successfully!")
        print(f"   Number of classes: {len(crf.classes_)}")
        print(f"   Classes: {sorted(list(crf.classes_))[:10]}...")
        
        # Simple evaluation
        if len(all_sents) > 5:
            from sklearn.model_selection import train_test_split
            from sklearn_crfsuite import metrics
            
            # Split for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42
            )
            
            # Train eval model
            eval_crf = CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True
            )
            eval_crf.fit(X_train, y_train)
            
            # Predict
            y_pred = eval_crf.predict(X_test)
            
            # Calculate accuracy
            correct = 0
            total = 0
            for true_seq, pred_seq in zip(y_test, y_pred):
                for true, pred in zip(true_seq, pred_seq):
                    if true == pred:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            print(f"📈 Evaluation Accuracy: {accuracy:.2%}")
        
        # Save model
        print(f"💾 Saving model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(crf, model_path, compress=3)
        
        # Verify save
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"✅ Model saved: {size:,} bytes ({size/1024:.1f} KB)")
            
            # Test load
            loaded_model = joblib.load(model_path)
            print(f"   Can load: {loaded_model is not None}")
            print(f"   Has predict: {hasattr(loaded_model, 'predict')}")
        else:
            print("❌ Failed to save model")
        
        return crf
        
    except ImportError:
        print("❌ sklearn_crfsuite not installed. Please install:")
        print("   pip install sklearn-crfsuite")
        return None
    except Exception as e:
        print(f"❌ CRF training error: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_hmm_model(train_sents=None, dev_sents=None, model_path="saved_models/hmm_model.joblib"):
    """Huấn luyện HMM model (không OOV)"""
    print("\n" + "="*60)
    print("🤖 TRAINING HMM MODEL")
    print("="*60)
    
    # Sử dụng dữ liệu có sẵn hoặc tạo synthetic
    if train_sents is None or dev_sents is None:
        print("⚠️ No training data provided, using synthetic data")
        data = create_synthetic_data()
        # Split thành train và dev
        split_idx = int(len(data) * 0.8)
        train_sents = data[:split_idx]
        dev_sents = data[split_idx:]
    else:
        print(f"✅ Using provided data: {len(train_sents)} train, {len(dev_sents)} dev")
    
    # Kết hợp train và dev
    combined_data = train_sents + dev_sents
    
    print(f"📊 Total sentences for training: {len(combined_data)}")
    
    try:
        from nltk.tag import hmm
        
        print("🧠 Training HMM model...")
        trainer = hmm.HiddenMarkovModelTrainer()
        hmm_tagger = trainer.train_supervised(combined_data)
        
        print("✅ HMM model trained successfully!")
        
        # Tạo vocab
        vocab = set()
        for sent in combined_data:
            for word, _ in sent:
                vocab.add(word.lower())
        
        print(f"   Vocabulary size: {len(vocab)}")
        print(f"   Number of states (tags): {len(hmm_tagger._states)}")
        
        # Lưu model và vocab
        save_data = (hmm_tagger, vocab)
        
        print(f"💾 Saving model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(save_data, model_path, compress=3)
        
        # Verify save
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"✅ Model saved: {size:,} bytes ({size/1024:.1f} KB)")
        else:
            print("❌ Failed to save model")
        
        return hmm_tagger
        
    except ImportError:
        print("❌ nltk not installed. Please install:")
        print("   pip install nltk")
        return None
    except Exception as e:
        print(f"❌ HMM training error: {e}")
        return None

def train_hmm_oov_model(train_sents=None, dev_sents=None, model_path="saved_models/hmm_oov_model.joblib"):
    """Huấn luyện HMM model với xử lý OOV"""
    print("\n" + "="*60)
    print("🤖 TRAINING HMM WITH OOV HANDLING")
    print("="*60)
    
    # Sử dụng dữ liệu có sẵn hoặc tạo synthetic
    if train_sents is None or dev_sents is None:
        print("⚠️ No training data provided, using synthetic data")
        data = create_synthetic_data()
        # Split thành train và dev
        split_idx = int(len(data) * 0.8)
        train_sents = data[:split_idx]
        dev_sents = data[split_idx:]
    else:
        print(f"✅ Using provided data: {len(train_sents)} train, {len(dev_sents)} dev")
    
    # Kết hợp train và dev
    combined_data = train_sents + dev_sents
    
    print(f"📊 Original sentences: {len(combined_data)}")
    
    # Tạo vocab
    vocab = set()
    for sent in combined_data:
        for word, _ in sent:
            vocab.add(word.lower())
    
    print(f"   Vocabulary size: {len(vocab)}")
    
    # OOV token
    OOV_TOKEN = "__OOV__"
    
    # Augment với OOV samples
    print("🔄 Augmenting with OOV samples...")
    augmented_data = combined_data.copy()
    
    # Thêm sentences với OOV tokens
    num_oov_samples = min(300, len(combined_data) * 2)  # Thêm OOV samples
    for i in range(0, num_oov_samples, 5):
        # Tạo sentence với 1-5 OOV tokens
        num_oov_in_sent = (i % 5) + 1
        oov_sentence = [(OOV_TOKEN, "NOUN") for _ in range(num_oov_in_sent)]
        augmented_data.append(oov_sentence)
    
    print(f"   Total sentences after OOV augmentation: {len(augmented_data)}")
    
    try:
        from nltk.tag import hmm
        
        print("🧠 Training HMM-OOV model...")
        trainer = hmm.HiddenMarkovModelTrainer()
        hmm_tagger = trainer.train_supervised(augmented_data)
        
        print("✅ HMM-OOV model trained successfully!")
        print(f"   OOV token: {OOV_TOKEN}")
        print(f"   Number of states (tags): {len(hmm_tagger._states)}")
        
        # Lưu model và vocab
        save_data = (hmm_tagger, vocab, OOV_TOKEN)
        
        print(f"💾 Saving model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(save_data, model_path, compress=3)
        
        # Verify save
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"✅ Model saved: {size:,} bytes ({size/1024:.1f} KB)")
        else:
            print("❌ Failed to save model")
        
        return hmm_tagger
        
    except ImportError:
        print("❌ nltk not installed. Please install:")
        print("   pip install nltk")
        return None
    except Exception as e:
        print(f"❌ HMM-OOV training error: {e}")
        return None

def train_all_models(data_dir=None):
    """Train tất cả models"""
    print("🚀 TRAINING ALL POS TAGGING MODELS")
    print("="*70)
    
    train_data = None
    dev_data = None
    
    # Load data nếu có
    if data_dir and os.path.exists(data_dir):
        try:
            from data_loader.loader import load_conllu_data
            
            print(f"📂 Loading data from {data_dir}...")
            train_data = load_conllu_data(os.path.join(data_dir, "en_ewt-ud-train.conllu"))
            dev_data = load_conllu_data(os.path.join(data_dir, "en_ewt-ud-dev.conllu"))
            
            print(f"✅ Loaded: {len(train_data)} train, {len(dev_data)} dev sentences")
            
        except ImportError:
            print("⚠️ Could not load data_loader, using synthetic data")
        except Exception as e:
            print(f"⚠️ Error loading data: {e}, using synthetic data")
    
    # Train CRF model
    crf_model = train_crf_model(train_data, dev_data, "saved_models/crf_model.joblib")
    
    # Train HMM model
    hmm_model = train_hmm_model(train_data, dev_data, "saved_models/hmm_model.joblib")
    
    # Train HMM-OOV model
    hmm_oov_model = train_hmm_oov_model(train_data, dev_data, "saved_models/hmm_oov_model.joblib")
    
    print("\n" + "="*70)
    print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📋 Models saved to 'saved_models/' directory:")
    print("   • crf_model.joblib - Conditional Random Fields")
    print("   • hmm_model.joblib - Hidden Markov Model")
    print("   • hmm_oov_model.joblib - HMM with OOV handling")
    
    print("\n🚀 Next steps:")
    print("   1. Run: streamlit run streamlit_app.py")
    print("   2. Select different models from the sidebar")
    print("   3. Enter text to see POS tagging results")
    
    return {
        'crf': crf_model,
        'hmm': hmm_model,
        'hmm_oov': hmm_oov_model
    }

def test_trained_models():
    """Test tất cả models đã train"""
    print("\n🧪 TESTING TRAINED MODELS")
    print("="*60)
    
    test_sentences = [
        "I'm the one who shall grid",
        "The quick brown fox jumps",
        "Google is a technology company",
        "She will arrive tomorrow at 3 PM"
    ]
    
    # Test CRF
    print("\n🔍 Testing CRF Model:")
    crf_path = "saved_models/crf_model.joblib"
    if os.path.exists(crf_path):
        try:
            from models.crf_model import CRFTagger
            tagger = CRFTagger(crf_path)
            
            for sentence in test_sentences[:2]:
                print(f"\n  '{sentence}'")
                tokens = sentence.split()
                tags = tagger.tag(tokens)
                for token, tag in zip(tokens, tags):
                    print(f"    '{token:15}' -> {tag}")
        except Exception as e:
            print(f"  ❌ CRF test error: {e}")
    else:
        print("  ❌ CRF model not found")
    
    # Test HMM
    print("\n🔍 Testing HMM Model:")
    hmm_path = "saved_models/hmm_model.joblib"
    if os.path.exists(hmm_path):
        try:
            from models.hmm_model import HMMTagger
            tagger = HMMTagger(hmm_path)
            
            for sentence in test_sentences[:2]:
                print(f"\n  '{sentence}'")
                tokens = sentence.split()
                tags = tagger.tag(tokens)
                for token, tag in zip(tokens, tags):
                    print(f"    '{token:15}' -> {tag}")
        except Exception as e:
            print(f"  ❌ HMM test error: {e}")
    else:
        print("  ❌ HMM model not found")
    
    # Test HMM-OOV
    print("\n🔍 Testing HMM-OOV Model:")
    hmm_oov_path = "saved_models/hmm_oov_model.joblib"
    if os.path.exists(hmm_oov_path):
        try:
            from models.hmm_oov_model import HMMOOVTagger
            tagger = HMMOOVTagger(hmm_oov_path)
            
            for sentence in test_sentences:
                print(f"\n  '{sentence}'")
                tokens = sentence.split()
                tags = tagger.tag(tokens)
                for token, tag in zip(tokens, tags):
                    oov_marker = " (OOV)" if tagger.vocab and token.lower() not in tagger.vocab else ""
                    print(f"    '{token:15}' -> {tag}{oov_marker}")
        except Exception as e:
            print(f"  ❌ HMM-OOV test error: {e}")
    else:
        print("  ❌ HMM-OOV model not found")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train POS tagging models')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--test', action='store_true', help='Test trained models')
    
    args = parser.parse_args()
    
    if args.test:
        test_trained_models()
    else:
        train_all_models(args.data_dir)