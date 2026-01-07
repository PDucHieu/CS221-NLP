import pickle
from sklearn_crfsuite import metrics
from preprocessing.load_data import load_conllu_data

test = load_conllu_data("data/raw/en_ewt-ud-test.conllu")

with open("saved_models/hmm_oov.pkl", "rb") as f:
    model, vocab = pickle.load(f)

def replace_oov(seq):
    return [w if w in vocab else "__OOV__" for w in seq]

y_true = []
y_pred = []

for sent in test:
    words = replace_oov([w for w, _ in sent])
    tags = [t for _, t in sent]
    pred = [t for _, t in model.tag(words)]
    y_true.append(tags)
    y_pred.append(pred)

print(metrics.flat_classification_report(y_true, y_pred))
