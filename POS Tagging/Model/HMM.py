import pickle

class HMMTagger:
    def __init__(self):
        with open("saved_models/hmm.pkl", "rb") as f:
            self.model = pickle.load(f)

    def tag(self, tokens):
        return [t for _, t in self.model.tag(tokens)]
