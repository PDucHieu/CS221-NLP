import pickle

class HMMOOVTagger:
    def __init__(self):
        with open("saved_models/hmm_oov.pkl", "rb") as f:
            self.model, self.vocab = pickle.load(f)

    def tag(self, tokens):
        tokens = [w if w in self.vocab else "__OOV__" for w in tokens]
        return [t for _, t in self.model.tag(tokens)]
