# preprocessing/text_utils.py
import random

def is_float(s):
    """Kiểm tra xem string có phải float không"""
    try:
        float(s)
        return True
    except ValueError:
        return False

def replace_oov_words(sent_words, known_vocab, oov_token="__OOV__"):
    """Thay thế từ ngoài từ điển bằng OOV token"""
    return [w if w in known_vocab else oov_token for w in sent_words]

def augment_training_with_oov(train_sents, oov_token="__OOV__", num_samples=200):
    """Bổ sung dữ liệu OOV cho training"""
    random_tags = ['NOUN', 'PROPN', 'X', 'NUM', 'INTJ', 'ADJ']
    oov_augmented = []
    for _ in range(num_samples):
        tag = random.choice(random_tags)
        oov_augmented.append([(oov_token, tag)])
    return train_sents + oov_augmented