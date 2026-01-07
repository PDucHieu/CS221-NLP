import streamlit as st

from models.crf_model import CRFTagger
from models.hmm_model import HMMTagger
from models.hmm_oov_model import HMMOOVTagger

st.title("POS Tagging Demo (NLP)")

sentence = st.text_input("Enter an English sentence:")

model_name = st.selectbox(
    "Choose model:",
    ["CRF", "HMM", "HMM + OOV"]
)

if st.button("Tag POS"):
    tokens = sentence.split()

    if model_name == "CRF":
        tagger = CRFTagger()
    elif model_name == "HMM":
        tagger = HMMTagger()
    else:
        tagger = HMMOOVTagger()

    tags = tagger.tag(tokens)

    st.subheader("Result")
    for w, t in zip(tokens, tags):
        st.write(f"**{w}** → `{t}`")
