import pickle
from nltk.tag import hmm

from preprocessing.load_data import load_conllu_data

OOV = "__OOV__"

train = load_conllu_data("data/raw/en_ewt-ud-train.conllu")
dev   = load_conllu_data("data/raw/en_ewt-ud-dev.conllu")

data = train + dev

vocab = set(word for sent in data for word, _ in sent)

augmented = data + [[(OOV, "NOUN")] for _ in range(300)]

trainer = hmm.HiddenMarkovModelTrainer()
model = trainer.train_supervised(augmented)

with open("saved_models/hmm_oov.pkl", "wb") as f:
    pickle.dump((model, vocab), f)

print("HMM + OOV trained and saved.")
