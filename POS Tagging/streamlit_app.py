# streamlit_app.py
import streamlit as st
import sys
import os
import pandas as pd
from collections import Counter

# Thêm đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Cấu hình trang
st.set_page_config(
    page_title="POS Tagging Demo",
    page_icon="🔠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .tag-box {
        display: inline-block;
        padding: 8px 16px;
        margin: 4px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .tag-box:hover {
        transform: translateY(-2px);
    }
    .result-container {
        background: linear-gradient(135deg, #F3F4F6 0%, #FFFFFF 100%);
        padding: 25px;
        border-radius: 12px;
        margin-top: 25px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #3B82F6;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1E40AF, #3B82F6);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Import models với try-except
try:
    from models.crf_model import CRFTagger
    CRF_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ CRF Model Import Error: {e}")
    CRF_AVAILABLE = False

try:
    from models.hmm_model import HMMTagger
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from models.hmm_oov_model import HMMOOVTagger
    HMM_OOV_AVAILABLE = True
except ImportError:
    HMM_OOV_AVAILABLE = False

# Fallback tagger
class FallbackTagger:
    def __init__(self):
        self.name = "Rule-Based Fallback"
    
    def tag(self, tokens):
        tags = []
        for token in tokens:
            if not token:
                tags.append("X")
            elif token == "I":
                tags.append("PRON")
            elif token.lower() in ['the', 'a', 'an']:
                tags.append("DET")
            elif token.lower() in ['is', 'am', 'are', 'was', 'were']:
                tags.append("VERB")
            elif token.endswith('ing'):
                tags.append("VERB")
            elif token.endswith('ly'):
                tags.append("ADV")
            elif token[0].isupper():
                tags.append("PROPN")
            elif any(c.isdigit() for c in token):
                tags.append("NUM")
            else:
                tags.append("NOUN")
        return tags

def main():
    # Header
    st.markdown('<h1 class="main-header">🔠 Advanced POS Tagging Demo</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6B7280;">Professional Part-of-Speech Tagging with NLP Models</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Model selection
        available_models = []
        if CRF_AVAILABLE:
            available_models.append("CRF")
        if HMM_AVAILABLE:
            available_models.append("HMM")
        if HMM_OOV_AVAILABLE:
            available_models.append("HMM with OOV")
        
        if not available_models:
            available_models = ["Rule-Based"]
        
        model_choice = st.selectbox(
            "**Select Model:**",
            available_models,
            help="Choose the NLP model for POS tagging"
        )
        
        st.markdown("---")
        
        # Model status
        st.markdown("## 📊 Model Status")
        
        # Check model files
        model_files = {
            "CRF": "saved_models/crf_model.joblib",
            "HMM": "saved_models/hmm_model.joblib",
            "HMM with OOV": "saved_models/hmm_oov_model.joblib"
        }
        
        for model_name, file_path in model_files.items():
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                st.success(f"✅ **{model_name}**: {size:,} bytes")
            else:
                st.warning(f"⚠️ **{model_name}**: File not found")
        
        st.markdown("---")
        
        # Settings
        st.markdown("## 🎛️ Settings")
        tokenize_option = st.checkbox("Smart Tokenization", value=True)
        show_details = st.checkbox("Show Detailed Analysis", value=True)
        
        st.markdown("---")
        
        # Information
        with st.expander("📚 Model Information"):
            st.markdown("""
            **🎯 CRF (Conditional Random Fields)**
            - Uses contextual features
            - Considers word neighbors
            - Best for accuracy
            
            **📊 HMM (Hidden Markov Model)**
            - Statistical approach
            - Fast inference
            - Memory efficient
            
            **🆕 HMM with OOV Handling**
            - Handles unknown words
            - Robust for diverse text
            """)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("## 📝 Text Input")
        
        # Example sentences
        examples = {
            "General": [
                "I love natural language processing.",
                "The quick brown fox jumps over the lazy dog.",
                "She will be arriving at 3 PM tomorrow."
            ],
            "Technical": [
                "The transformer model uses attention mechanisms.",
                "Python 3.11 includes performance improvements.",
                "The algorithm achieved 95% accuracy."
            ],
            "Business": [
                "Our revenue increased by 20% this quarter.",
                "Please review the attached documents.",
                "The meeting will discuss marketing strategy."
            ]
        }
        
        # Tabs
        tab1, tab2 = st.tabs(["✏️ Custom Input", "📋 Examples"])
        
        with tab1:
            input_text = st.text_area(
                "Enter your text:",
                height=150,
                placeholder="Type or paste your text here...",
                help="Text to analyze for part-of-speech tags"
            )
        
        with tab2:
            category = st.selectbox("Category:", list(examples.keys()))
            selected_example = st.selectbox("Select example:", examples[category])
            if st.button("Use This Example", key="load_example"):
                input_text = selected_example
                st.rerun()
    
    with col2:
        st.markdown("## 🏷️ Tag Legend")
        
        tag_colors = {
            'NOUN': '#3B82F6',    # Blue
            'VERB': '#10B981',    # Green
            'ADJ': '#F59E0B',     # Amber
            'ADV': '#8B5CF6',     # Violet
            'PROPN': '#EF4444',   # Red
            'DET': '#EC4899',     # Pink
            'PRON': '#14B8A6',    # Teal
            'CONJ': '#F97316',    # Orange
            'ADP': '#6366F1',     # Indigo
            'NUM': '#84CC16',     # Lime
            'PUNCT': '#64748B',   # Slate
            'AUX': '#A855F7',     # Purple
            'X': '#9CA3AF',       # Gray
        }
        
        # Hiển thị common tags
        common_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'DET']
        cols = st.columns(3)
        for idx, tag in enumerate(common_tags):
            with cols[idx % 3]:
                color = tag_colors.get(tag, '#6B7280')
                st.markdown(
                    f'<div style="background:{color}; color:white; padding:6px 10px; '
                    f'border-radius:6px; text-align:center; font-size:12px; font-weight:600;">{tag}</div>',
                    unsafe_allow_html=True
                )
        
        with st.expander("All Tags"):
            cols = st.columns(4)
            for idx, (tag, color) in enumerate(tag_colors.items()):
                with cols[idx % 4]:
                    st.markdown(
                        f'<div style="background:{color}; color:white; padding:4px 8px; '
                        f'border-radius:4px; text-align:center; font-size:11px; margin:2px;">{tag}</div>',
                        unsafe_allow_html=True
                    )
    
    # Process button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_btn = st.button(
            f"🚀 Analyze with {model_choice}",
            type="primary",
            use_container_width=True,
            disabled=not input_text.strip() if 'input_text' in locals() else True
        )
    
    # Process khi có input
    if 'input_text' in locals() and process_btn and input_text.strip():
        with st.spinner(f"🧠 Analyzing with {model_choice}..."):
            progress_bar = st.progress(0)
            
            try:
                # Tokenization
                progress_bar.progress(20)
                if tokenize_option:
                    try:
                        import nltk
                        nltk.download('punkt', quiet=True)
                        tokens = nltk.word_tokenize(input_text)
                    except:
                        tokens = input_text.strip().split()
                else:
                    tokens = input_text.strip().split()
                
                # Load model
                progress_bar.progress(40)
                tagger = None
                
                if model_choice == "CRF" and CRF_AVAILABLE:
                    tagger = CRFTagger()
                    model_source = "CRF Model" if hasattr(tagger, 'model_loaded') and tagger.model_loaded else "CRF Fallback"
                elif model_choice == "HMM" and HMM_AVAILABLE:
                    tagger = HMMTagger()
                    model_source = "HMM Model"
                elif model_choice == "HMM with OOV" and HMM_OOV_AVAILABLE:
                    tagger = HMMOOVTagger()
                    model_source = "HMM-OOV Model"
                else:
                    tagger = FallbackTagger()
                    model_source = "Rule-Based Fallback"
                
                # Get predictions
                progress_bar.progress(70)
                tags = tagger.tag(tokens)
                
                progress_bar.progress(90)
                
                # Display results
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                # Header
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"## 📊 Analysis Results")
                    st.markdown(f"**Model:** {model_choice} | **Source:** {model_source} | **Tokens:** {len(tokens)}")
                with col2:
                    if model_source == "CRF Model":
                        st.success("✅ Using trained model")
                    else:
                        st.info("ℹ️ Using fallback/rules")
                
                # Visual tagged sentence
                st.markdown("**Tagged Sentence:**")
                
                html_tags = []
                for token, tag in zip(tokens, tags):
                    color = tag_colors.get(tag, '#9CA3AF')
                    html_tags.append(
                        f'<span style="display:inline-block; padding:8px 14px; margin:4px; '
                        f'border-radius:8px; background:{color}; color:white; font-weight:600; '
                        f'box-shadow:0 2px 4px rgba(0,0,0,0.1);">'
                        f'{token}<br><small style="font-size:0.8em; opacity:0.9;">{tag}</small>'
                        f'</span>'
                    )
                
                st.markdown(
                    '<div style="line-height:2.2; margin:20px 0; padding:15px; '
                    'background:white; border-radius:10px; border:1px solid #E5E7EB;">' + 
                    ' '.join(html_tags) + 
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Detailed table
                if show_details:
                    st.markdown("**Detailed Breakdown:**")
                    df = pd.DataFrame({
                        'Token': tokens,
                        'POS Tag': tags,
                        'Length': [len(t) for t in tokens]
                    })
                    
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Token": st.column_config.TextColumn("Token", width="medium"),
                            "POS Tag": st.column_config.TextColumn("POS Tag", width="small"),
                            "Length": st.column_config.NumberColumn("Length", width="small")
                        }
                    )
                
                # Statistics
                st.markdown("**📈 Statistics**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Total Tokens", len(tokens))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    unique_tags = len(set(tags))
                    st.metric("Unique POS Tags", unique_tags)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    if tags:
                        most_common = Counter(tags).most_common(1)[0]
                        st.metric("Most Common", most_common[0], f"{most_common[1]} occ")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Tag distribution
                if show_details and tags:
                    st.markdown("**📊 Tag Distribution**")
                    tag_counts = Counter(tags)
                    tag_df = pd.DataFrame({
                        'POS Tag': list(tag_counts.keys()),
                        'Count': list(tag_counts.values()),
                        'Percentage': [f"{(count/len(tags)*100):.1f}%" for count in tag_counts.values()]
                    }).sort_values('Count', ascending=False)
                    
                    st.dataframe(tag_df, use_container_width=True, hide_index=True)
                
                progress_bar.progress(100)
                st.success("✅ Analysis completed!")
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)[:200]}")
                st.info("💡 Tip: Check console for detailed error messages")
    
    # Guide khi chưa có input
    elif 'input_text' not in locals() or not input_text.strip():
        st.markdown("---")
        with st.container():
            st.markdown("## 🚀 Getting Started")
            cols = st.columns(3)
            
            with cols[0]:
                st.info("### 1. Enter Text\nType or paste text, or select an example.")
            
            with cols[1]:
                st.info("### 2. Choose Model\nSelect a model from the sidebar.")
            
            with cols[2]:
                st.info("### 3. Analyze\nClick Analyze to see POS tagging results.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.9rem; padding: 20px;">
        <p style="margin: 5px 0;">🔠 <strong>POS Tagging Demo</strong> | Built with Streamlit</p>
        <p style="margin: 5px 0; font-size: 0.8rem;">Supports CRF, HMM, and HMM-OOV Models</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()