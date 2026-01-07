import joblib
from preprocessing.features import extract_features

class CRFTagger:
    def __init__(self):
        self.model = joblib.load("saved_models/crf.joblib")

    def tag(self, tokens):
        sent = [(w, "") for w in tokens]
        X = [extract_features(sent)]
        return self.model.predict(X)[0]
