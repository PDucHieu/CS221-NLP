# data_loader/loader.py
import os
from conllu import parse_incr
import numpy as np

def load_conllu_data(filepath):
    """Load data từ file conllu format"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    
    data = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for sentence in parse_incr(f):
                sent = []
                for token in sentence:
                    if isinstance(token["id"], int):
                        word = token["form"]
                        pos = token["upostag"]
                        if word and pos:
                            sent.append((word, pos))
                if sent:
                    data.append(sent)
    except Exception as e:
        print(f"Lỗi khi đọc file {filepath}: {e}")
        raise
    
    return data

def validate_data(sents, dataset_name=""):
    """Kiểm tra chất lượng dữ liệu"""
    print(f"\n=== Kiểm tra dữ liệu {dataset_name} ===")
    empty_sents = 0
    invalid_tokens = 0
    
    for i, sent in enumerate(sents):
        if not sent:
            empty_sents += 1
            continue
        for word, pos in sent:
            if not word or not pos or word.strip() == '' or pos.strip() == '':
                invalid_tokens += 1
                print(f"Warning: Token không hợp lệ tại câu {i}: word='{word}', pos='{pos}'")
    
    print(f"Tổng số câu: {len(sents)}")
    print(f"Câu rỗng: {empty_sents}")
    print(f"Token không hợp lệ: {invalid_tokens}")
    
    sent_lengths = [len(sent) for sent in sents if sent]
    if sent_lengths:
        print(f"Độ dài câu trung bình: {np.mean(sent_lengths):.1f}")
        print(f"Độ dài câu min/max: {min(sent_lengths)}/{max(sent_lengths)}")

def clean_data(sents):
    """Làm sạch dữ liệu"""
    cleaned = []
    for sent in sents:
        if not sent:
            continue
        clean_sent = []
        for word, pos in sent:
            if word and pos and word.strip() and pos.strip():
                clean_sent.append((word.strip(), pos.strip()))
        if clean_sent:
            cleaned.append(clean_sent)
    return cleaned