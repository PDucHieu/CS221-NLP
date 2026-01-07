import string
import re
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

def word_shape(word):
    shape = re.sub(r"[A-Z]", "X", word)
    shape = re.sub(r"[a-z]", "x", shape)
    shape = re.sub(r"[0-9]", "d", shape)
    return shape

def char_ngrams(word, n=2, max_n=3):
    return [
        word[i:i+n]
        for i in range(len(word) - n + 1)
    ][:max_n]

def word2features(sent, i):
    word = sent[i][0]

    features = {
        "bias": 1.0,

        # lexical
        "word.lower": word.lower(),
        "word.len": len(word),

        # suffix / prefix
        "suffix1": word[-1:],
        "suffix2": word[-2:],
        "suffix3": word[-3:],
        "prefix1": word[:1],
        "prefix2": word[:2],
        "prefix3": word[:3],

        # casing
        "isupper": word.isupper(),
        "islower": word.islower(),
        "istitle": word.istitle(),

        # char types
        "isdigit": word.isdigit(),
        "has_digit": any(c.isdigit() for c in word),
        "has_punct": any(c in string.punctuation for c in word),

        # shape
        "shape": word_shape(word),

        # stopword
        "is_stopword": word.lower() in STOP_WORDS,

        # char ngrams
        "char_bigram": " ".join(char_ngrams(word, 2)),
        "char_trigram": " ".join(char_ngrams(word, 3)),
    }

    # context -1
    if i > 0:
        prev = sent[i - 1][0]
        features.update({
            "-1.word.lower": prev.lower(),
            "-1.istitle": prev.istitle(),
            "-1.shape": word_shape(prev),
        })
    else:
        features["BOS"] = True

    # context +1
    if i < len(sent) - 1:
        nxt = sent[i + 1][0]
        features.update({
            "+1.word.lower": nxt.lower(),
            "+1.istitle": nxt.istitle(),
            "+1.shape": word_shape(nxt),
        })
    else:
        features["EOS"] = True

    return features


def extract_features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def extract_labels(sent):
    return [label for _, label in sent]
