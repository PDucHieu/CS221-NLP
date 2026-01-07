import pickle
from nltk.tag import hmm

from preprocessing.load_data import load_conllu_data

train = load_conllu_data("data/raw/en_ewt-ud-train.conllu")
dev   = load_conllu_data("data/raw/en_ewt-ud-dev.conllu")

data = train + dev

trainer = hmm.HiddenMarkovModelTrainer()
model = trainer.train_supervised(data)

with open("saved_models/hmm.pkl", "wb") as f:
    pickle.dump(model, f)

print("HMM trained and saved.")
