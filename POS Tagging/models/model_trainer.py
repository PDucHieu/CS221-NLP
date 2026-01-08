# models/model_trainer.py
import joblib
from sklearn_crfsuite import CRF
from sklearn.model_selection import ParameterGrid, KFold
import numpy as np
from copy import deepcopy
import nltk
from nltk.tag import hmm

from data_loader.loader import load_conllu_data, clean_data
from preprocessing.features import extract_features, get_labels
from preprocessing.text_utils import augment_training_with_oov

def train_crf_model(train_sents, dev_sents, model_path="saved_models/crf_model.joblib"):
    """Huấn luyện CRF model"""
    print("Training CRF model...")
    
    # Kết hợp train và dev
    all_sents = train_sents + dev_sents
    
    # Trích xuất features
    X_all = [extract_features(s) for s in all_sents]
    y_all = [get_labels(s) for s in all_sents]
    
    # Hyperparameter tuning đơn giản
    param_grid = {
        'c1': [0.1, 1.0],
        'c2': [0.1, 1.0],
        'max_iterations': [100],
        'algorithm': ['lbfgs']
    }
    
    best_score = 0
    best_params = {}
    
    # Cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    for params in ParameterGrid(param_grid):
        print(f"Testing params: {params}")
        fold_scores = []
        
        for train_idx, val_idx in kf.split(X_all):
            X_train = [X_all[i] for i in train_idx]
            y_train = [y_all[i] for i in train_idx]
            X_val = [X_all[i] for i in val_idx]
            y_val = [y_all[i] for i in val_idx]
            
            crf = CRF(
                algorithm=params['algorithm'],
                c1=params['c1'],
                c2=params['c2'],
                max_iterations=params['max_iterations'],
                all_possible_transitions=True
            )
            
            crf.fit(X_train, y_train)
            y_pred = crf.predict(X_val)
            
            from sklearn_crfsuite import metrics
            score = metrics.flat_f1_score(y_val, y_pred, average='weighted')
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    print(f"Best params: {best_params}, Score: {best_score:.4f}")
    
    # Train final model
    final_crf = CRF(
        algorithm=best_params['algorithm'],
        c1=best_params['c1'],
        c2=best_params['c2'],
        max_iterations=best_params['max_iterations'],
        all_possible_transitions=True
    )
    
    final_crf.fit(X_all, y_all)
    
    # Save model
    joblib.dump(final_crf, model_path)
    print(f"CRF model saved to {model_path}")
    
    return final_crf

def train_hmm_model(train_sents, dev_sents, model_path="saved_models/hmm_model.joblib"):
    """Huấn luyện HMM model (không OOV)"""
    print("Training HMM model...")
    
    combined_data = train_sents + dev_sents
    
    trainer = hmm.HiddenMarkovModelTrainer()
    hmm_tagger = trainer.train_supervised(combined_data)
    
    # Thêm vocab vào model để lưu
    vocab = set(word for sent in combined_data for word, _ in sent)
    hmm_tagger.vocab = vocab
    
    joblib.dump(hmm_tagger, model_path)
    print(f"HMM model saved to {model_path}")
    
    return hmm_tagger

def train_hmm_oov_model(train_sents, dev_sents, model_path="saved_models/hmm_oov_model.joblib"):
    """Huấn luyện HMM model với xử lý OOV"""
    print("Training HMM OOV model...")
    
    # Xây dựng vocab
    combined_data = train_sents + dev_sents
    vocab = set(word for sent in combined_data for word, _ in sent)
    
    # Bổ sung dữ liệu OOV
    augmented_data = augment_training_with_oov(combined_data)
    
    trainer = hmm.HiddenMarkovModelTrainer()
    hmm_tagger = trainer.train_supervised(augmented_data)
    
    # Lưu vocab cùng model
    hmm_tagger.vocab = vocab
    
    joblib.dump(hmm_tagger, model_path)
    print(f"HMM OOV model saved to {model_path}")
    
    return hmm_tagger