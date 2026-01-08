# config.py
import os

# Đường dẫn dữ liệu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# File paths
TRAIN_FILE = os.path.join(DATA_DIR, "en_ewt-ud-train.conllu")
DEV_FILE = os.path.join(DATA_DIR, "en_ewt-ud-dev.conllu") 
TEST_FILE = os.path.join(DATA_DIR, "en_ewt-ud-test.conllu")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
CRF_MODEL_PATH = os.path.join(MODEL_DIR, "crf_model.joblib")
HMM_MODEL_PATH = os.path.join(MODEL_DIR, "hmm_model.joblib")
HMM_OOV_MODEL_PATH = os.path.join(MODEL_DIR, "hmm_oov_model.joblib")

# Cấu hình khác
OOV_TOKEN = "__OOV__"
AUGMENTATION_SAMPLES = 200
RANDOM_SEED = 42