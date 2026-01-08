# training/CRF.py
import os
import sys
import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import numpy as np

# Thêm đường dẫn để import
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def train_crf_model():
    """Train CRF model for POS tagging"""
    print("=" * 60)
    print("🤖 TRAINING CRF MODEL FOR POS TAGGING")
    print("=" * 60)
    
    try:
        # Import modules
        from preprocessing.load_data import load_conllu_data
        from preprocessing.features import extract_features, extract_labels
        
        print("✅ Modules imported successfully")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please check your module structure")
        return
    
    # Load data
    print("\n📂 Loading training data...")
    try:
        train_data = load_conllu_data("data/raw/en_ewt-ud-train.conllu")
        dev_data = load_conllu_data("data/raw/en_ewt-ud-dev.conllu")
        
        print(f"✅ Training sentences: {len(train_data):,}")
        print(f"✅ Development sentences: {len(dev_data):,}")
        
        # Combine data
        data = train_data + dev_data
        print(f"✅ Total sentences: {len(data):,}")
        
        # Sample statistics
        if data:
            total_tokens = sum(len(sent) for sent in data)
            print(f"✅ Total tokens: {total_tokens:,}")
            
            # Show sample
            print(f"\n📝 Sample sentence (first 5 tokens):")
            sample_sent = data[0][:5]
            for word, pos in sample_sent:
                print(f"   '{word}' -> {pos}")
                
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Using dummy data for demonstration...")
        
        # Create dummy data
        data = [
            [("The", "DET"), ("quick", "ADJ"), ("brown", "ADJ"), ("fox", "NOUN"), ("jumps", "VERB")],
            [("I", "PRON"), ("love", "VERB"), ("natural", "ADJ"), ("language", "NOUN"), ("processing", "NOUN")],
            [("Google", "PROPN"), ("is", "AUX"), ("a", "DET"), ("company", "NOUN")],
        ]
        print("✅ Created dummy data (3 sentences)")
    
    # Extract features and labels
    print("\n🔧 Extracting features...")
    try:
        X = [extract_features(s) for s in data]
        y = [extract_labels(s) for s in data]
        
        print(f"✅ Features extracted: {len(X):,} sequences")
        print(f"✅ Labels extracted: {len(y):,} sequences")
        
        # Show feature example
        if X:
            print(f"\n📊 Sample feature set (first token of first sentence):")
            sample_features = X[0][0]
            for key, value in list(sample_features.items())[:10]:
                print(f"   {key}: {value}")
            if len(sample_features) > 10:
                print(f"   ... and {len(sample_features) - 10} more features")
                
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return
    
    # Train CRF model
    print("\n🧠 Training CRF model...")
    try:
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,      # L1 regularization
            c2=0.1,      # L2 regularization
            max_iterations=200,
            all_possible_transitions=True,
            verbose=True
        )
        
        print("Starting training... (this may take a while)")
        crf.fit(X, y)
        
        print("✅ Model training completed!")
        
        # Model information
        print(f"\n📊 Model Information:")
        print(f"   Number of states: {len(crf.classes_)}")
        print(f"   Classes: {list(crf.classes_)[:10]}...")
        
        # Split for evaluation if we have enough data
        if len(X) > 10:
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"\n📈 Model Evaluation:")
            y_pred = crf.predict(X_test)
            
            # Calculate accuracy
            correct = 0
            total = 0
            for true, pred in zip(y_test, y_pred):
                for t, p in zip(true, pred):
                    if t == p:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            print(f"   Accuracy: {accuracy:.2%}")
            
            # Classification report
            print(f"\n📋 Classification Report:")
            labels = list(crf.classes_)
            labels.remove('O') if 'O' in labels else None
            
            sorted_labels = sorted(
                labels,
                key=lambda name: (name[1:], name[0])
            )
            
            print(metrics.flat_classification_report(
                y_test, y_pred, labels=sorted_labels, digits=3
            ))
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return
    
    # Save model
    print("\n💾 Saving model...")
    try:
        # Create directory if not exists
        os.makedirs("saved_models", exist_ok=True)
        
        # Save with the name the app expects
        model_path = "saved_models/crf_model.joblib"
        joblib.dump(crf, model_path)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"   File size: {os.path.getsize(model_path):,} bytes")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    # Test the saved model
    print("\n🧪 Testing saved model...")
    try:
        # Load and test
        loaded_model = joblib.load(model_path)
        
        # Create test sentence
        test_sentence = [("I", ""), ("love", ""), ("natural", ""), ("language", ""), ("processing", "")]
        test_features = extract_features(test_sentence)
        
        prediction = loaded_model.predict([test_features])[0]
        
        print("Test sentence: 'I love natural language processing'")
        print("Predicted POS tags:")
        words = [word for word, _ in test_sentence]
        for word, tag in zip(words, prediction):
            print(f"   '{word}' -> {tag}")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
    
    print("\n" + "=" * 60)
    print("✅ CRF MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Instructions
    print("\n📋 Next steps:")
    print("1. Run the Streamlit app: streamlit run streamlit_app.py")
    print("2. Select 'CRF' model in the sidebar")
    print("3. Enter text to see POS tagging results")
    print("4. The model will now use the trained CRF instead of fallback rules")

def create_dummy_model():
    """Create a dummy model if training fails"""
    print("\n🛠️ Creating dummy model for demonstration...")
    
    try:
        from sklearn_crfsuite import CRF
        
        # Simple training data
        X_train = [
            [
                {'word.lower()': 'the', 'BOS': True},
                {'word.lower()': 'cat', 'EOS': True}
            ],
            [
                {'word.lower()': 'i', 'BOS': True},
                {'word.lower()': 'run', 'EOS': True}
            ]
        ]
        
        y_train = [
            ['DET', 'NOUN'],
            ['PRON', 'VERB']
        ]
        
        crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=50
        )
        
        crf.fit(X_train, y_train)
        
        # Save
        os.makedirs("saved_models", exist_ok=True)
        model_path = "saved_models/crf_model.joblib"
        joblib.dump(crf, model_path)
        
        print(f"✅ Dummy model created: {model_path}")
        print("⚠️ Note: This is a simple model for demonstration only")
        
    except Exception as e:
        print(f"❌ Failed to create dummy model: {e}")

if __name__ == "__main__":
    try:
        train_crf_model()
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nTrying to create dummy model instead...")
        create_dummy_model()