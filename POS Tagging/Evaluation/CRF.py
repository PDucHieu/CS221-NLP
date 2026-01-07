import joblib
from sklearn_crfsuite import metrics

from preprocessing.load_data import load_conllu_data
from preprocessing.features import extract_features, extract_labels

test = load_conllu_data("data/raw/en_ewt-ud-test.conllu")

X_test = [extract_features(s) for s in test]
y_test = [extract_labels(s) for s in test]

crf = joblib.load("saved_models/crf.joblib")
y_pred = crf.predict(X_test)

print(metrics.flat_classification_report(y_test, y_pred))
