import streamlit as st
from models.hmm import HMMTagger
from models.hmm_oov import HMMOOVTagger
from models.crf import CRFTagger
from utils.preprocess import preprocess_sentence
from utils.visualization import render_pos_table

st.set_page_config(
    page_title="POS Tagging Application",
    layout="wide"
)

# ================== HEADER ==================
st.markdown(
    "<h1 style='color:#ff7f0e;'>Ứng Dụng POS Tagging 📚</h1>",
    unsafe_allow_html=True
)

st.markdown("""
### Giới Thiệu Ứng Dụng
- Ứng dụng NLP cho bài toán **Part-of-Speech Tagging**
- So sánh **HMM**, **HMM có xử lý OOV**, và **CRF**
- Hỗ trợ văn bản **tiếng Anh**
""")

st.divider()

# ================== MODEL SELECTION ==================
st.markdown("## 01) Chọn mô hình POS Tagging")

model_name = st.radio(
    "Chọn phương pháp:",
    ["HMM", "HMM-OOV", "CRF"],
    horizontal=True
)

# ================== INPUT ==================
st.markdown("## 02) Nhập dữ liệu")

input_type = st.radio(
    "Chọn cách nhập:",
    ["Câu", "File (.txt)"],
    horizontal=True
)

text = ""

if input_type == "Câu":
    text = st.text_area(
        "Nhập câu tiếng Anh:",
        "The quick brown fox jumps over the lazy dog."
    )
else:
    uploaded_file = st.file_uploader("Upload file .txt", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")

# ================== TAGGING ==================
st.divider()

if st.button("Thực hiện POS Tagging"):
    if not text.strip():
        st.warning("Vui lòng nhập văn bản.")
    else:
        tokens = preprocess_sentence(text)

        if model_name == "HMM":
            model = HMMTagger()
        elif model_name == "HMM-OOV":
            model = HMMOOVTagger()
        else:
            model = CRFTagger()

        pos_tags = model.predict(tokens)

        st.markdown("## Kết quả POS Tagging")
        render_pos_table(tokens, pos_tags)

# ================== SIDEBAR ==================
st.sidebar.header("Thông tin mô hình")

if model_name == "HMM":
    st.sidebar.markdown("""
    **Hidden Markov Model**
    - Generative model
    - Giả định Markov bậc 1
    - Dùng Viterbi decoding
    """)
elif model_name == "HMM-OOV":
    st.sidebar.markdown("""
    **HMM + OOV Handling**
    - Thêm xử lý từ chưa xuất hiện
    - Smoothing / suffix features
    """)
else:
    st.sidebar.markdown("""
    **Conditional Random Fields**
    - Discriminative sequence model
    - Sử dụng feature ngữ cảnh
    - Hiệu quả cao cho POS Tagging
    """)
