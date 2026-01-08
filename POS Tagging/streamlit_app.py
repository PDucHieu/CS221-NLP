# app.py
import streamlit as st
import sys
import os

# Thêm đường dẫn để import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import models
try:
    from models.crf_model import CRFTagger
    from models.hmm_model import HMMTagger
    from models.hmm_oov_model import HMMOOVTagger
    st.success("✅ Models imported successfully!")
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info("Please check the module structure")

# CSS styling
st.set_page_config(
    page_title="POS Tagging Demo",
    page_icon="🔠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tag-box {
        display: inline-block;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 14px;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔠 POS Tagging Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.selectbox(
            "Select Model:",
            ["CRF", "HMM", "HMM with OOV Handling"],
            index=0
        )
        
        st.markdown("---")
        st.header("ℹ️ About Models")
        st.markdown("""
        **CRF**: Conditional Random Fields  
        - Uses context features
        - Best accuracy
        
        **HMM**: Hidden Markov Model  
        - Statistical approach
        - Fast inference
        
        **HMM with OOV**:  
        - Handles unknown words
        - Uses OOV token replacement
        """)
        
        # Model info
        with st.expander("Model Details"):
            st.info(f"Selected: {model_choice}")
            st.write("This demo shows Part-of-Speech tagging using different NLP models.")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Text")
        
        # Example sentences
        examples = [
            "I love natural language processing.",
            "The quick brown fox jumps over the lazy dog.",
            "Google's new AI model is amazing!",
            "She will be arriving at 3 PM tomorrow."
        ]
        
        # Text input with examples
        example_choice = st.selectbox("Or choose an example:", 
                                    ["Custom input"] + examples)
        
        if example_choice == "Custom input":
            input_text = st.text_area("Enter your sentence:", 
                                    height=150,
                                    placeholder="Type your sentence here...")
        else:
            input_text = st.text_area("Enter your sentence:", 
                                    value=example_choice,
                                    height=150)
    
    with col2:
        st.subheader("🏷️ POS Tag Legend")
        
        tag_colors = {
            'NOUN': '#FF6B6B',    # Red
            'VERB': '#4ECDC4',    # Teal
            'ADJ': '#FFD166',     # Yellow
            'ADV': '#06D6A0',     # Green
            'PRON': '#118AB2',    # Blue
            'DET': '#EF476F',     # Pink
            'ADP': '#7B68EE',     # Purple
            'CONJ': '#20B2AA',    # Light Sea Green
            'PROPN': '#FF8C00',   # Dark Orange
            'NUM': '#9ACD32',     # Yellow Green
        }
        
        for tag, color in list(tag_colors.items())[:6]:  # Show first 6
            st.markdown(f'<span class="tag-box" style="background-color:{color};">{tag}</span>', 
                       unsafe_allow_html=True)
        
        with st.expander("See all tags"):
            for tag, color in tag_colors.items():
                st.markdown(f'<span class="tag-box" style="background-color:{color};margin:2px;">{tag}</span>', 
                           unsafe_allow_html=True)
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_btn = st.button("🚀 Tag POS", type="primary", use_container_width=True)
    
    if process_btn and input_text.strip():
        # Initialize model
        with st.spinner(f"Processing with {model_choice}..."):
            try:
                tokens = input_text.strip().split()
                
                if model_choice == "CRF":
                    tagger = CRFTagger()
                elif model_choice == "HMM":
                    tagger = HMMTagger()
                else:  # HMM with OOV
                    tagger = HMMOOVTagger()
                
                # Get predictions
                tags = tagger.tag(tokens)
                
                # Display results
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.subheader("📊 Results")
                
                # Create visualization
                st.markdown("**Tagged Sentence:**")
                
                html_tags = []
                for token, tag in zip(tokens, tags):
                    # Get color for tag (use first 4 chars for matching)
                    tag_key = tag[:4] if len(tag) > 4 else tag
                    color = tag_colors.get(tag_key, '#999999')
                    
                    html_tags.append(
                        f'<span class="tag-box" style="background-color:{color};color:white;">'
                        f'{token}<br><small>{tag}</small>'
                        f'</span>'
                    )
                
                st.markdown(
                    '<div style="line-height:2.5; margin:15px 0;">' + 
                    ' '.join(html_tags) + 
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Display as table
                st.markdown("**Detailed Breakdown:**")
                import pandas as pd
                df = pd.DataFrame({
                    'Token': tokens,
                    'POS Tag': tags
                })
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Statistics
                st.markdown("**Statistics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tokens", len(tokens))
                with col2:
                    st.metric("Unique POS Tags", len(set(tags)))
                with col3:
                    if tags:
                        from collections import Counter
                        most_common = Counter(tags).most_common(1)[0][0]
                        st.metric("Most Common Tag", most_common)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure model files exist in the 'saved_models' folder.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>POS Tagging Demo • Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()