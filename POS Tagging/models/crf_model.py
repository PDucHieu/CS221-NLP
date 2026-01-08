import sys
import os

# Import từ các module của bạn
# Giả sử bạn có structure:
# project/
#   ├── preprocessing/
#   ├── dataload/
#   ├── models/
#   └── saved_models/

sys.path.append('..')

# Import các module của bạn
from models.crf_model import CRFTagger
# Import preprocessing và dataload của bạn
# from preprocessing.your_preprocessor import Preprocessor
# from dataload.your_dataloader import DataLoader

def main():
    # 1. Khởi tạo tagger
    print("Initializing CRF Tagger...")
    tagger = CRFTagger()
    
    # 2. Test với câu mẫu
    test_sentences = [
        ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["I", "love", "natural", "language", "processing"],
        ["Apple", "is", "looking", "at", "buying", "U.K.", "startup", "for", "$1", "billion"]
    ]
    
    for i, tokens in enumerate(test_sentences):
        print(f"\n{'='*50}")
        print(f"Sentence {i+1}: {' '.join(tokens)}")
        print('-'*50)
        
        # Tag câu
        tags = tagger.tag(tokens)
        
        # Hiển thị kết quả
        for token, tag in zip(tokens, tags):
            print(f"{token:15} -> {tag}")
    
    # 3. Test với file model
    print(f"\n{'='*50}")
    print("Model status:")
    if tagger.model:
        print("✅ CRF model is loaded and ready")
    else:
        print("❌ CRF model failed to load")

if __name__ == "__main__":
    main()