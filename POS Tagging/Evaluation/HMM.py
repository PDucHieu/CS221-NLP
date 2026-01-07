import pickle
from sklearn_crfsuite import metrics
from preprocessing.load_data import load_conllu_data

test = load_conllu_data("data/raw/en_ewt-ud-test.conllu")

with open("saved_models/hmm.pkl", "rb") as f:
    model = pickle.load(f)

y_true = [[t for _, t in s] for s in test]
y_pred = [[t for _, t in model.tag([w for w, _ in s])] for s in test]

print(metrics.flat_classification_report(y_true, y_pred))
