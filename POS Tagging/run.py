# run.py
import os
import sys

# Đảm bảo import đúng thứ tự
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import các module theo thứ tự để tránh circular import
print("🔧 Setting up environment...")

# 1. Import preprocessing trước (nó không import models)
try:
    import preprocessing.features
    print("✅ preprocessing.features imported")
except Exception as e:
    print(f"⚠️ preprocessing.features: {e}")

# 2. Import models
try:
    # Test import CRF model
    from models.crf_model import CRFTagger
    print("✅ CRFTagger imported successfully")
    
    # Test tạo instance
    test_tagger = CRFTagger()
    print(f"   Model loaded: {test_tagger._model_loaded}")
    
except Exception as e:
    print(f"❌ CRF model import failed: {e}")
    import traceback
    traceback.print_exc()

# 3. Chạy app
print("\n🚀 Starting Streamlit app...")
import subprocess

# Chạy với environment đã setup
env = os.environ.copy()
env["PYTHONPATH"] = current_dir + ":" + env.get("PYTHONPATH", "")

subprocess.run([
    "streamlit", "run", "app.py",
    "--server.address", "0.0.0.0",
    "--server.port", "8501",
    "--theme.base", "light"
], env=env)