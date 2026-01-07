import joblib
import sklearn_crfsuite

from preprocessing.load_data import load_conllu_data
from preprocessing.features import extract_features, extract_labels

train = load_conllu_data("data/raw/en_ewt-ud-train.conllu")
dev   = load_conllu_data("data/raw/en_ewt-ud-dev.conllu")

data = train + dev

X = [extract_features(s) for s in data]
y = [extract_labels(s) for s in data]

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=200,
    all_possible_transitions=True
)

crf.fit(X, y)
joblib.dump(crf, "saved_models/crf.joblib")

print("CRF trained and saved.")
