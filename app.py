import streamlit as st
import torch
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from model_code import (
    DeepPersonDarkTriad, 
    encode_words, 
    encode_chars, 
    get_nrc_features_per_token,
    get_pos_features,
    get_readability_features,
    get_empath_features,
    clean_text,
    tokenize,
    MAX_SEQ_LEN,
    MAX_CHAR_LEN,
    EMB_DIM,
    CHAR_EMB_DIM,
    CHAR_OUT_CHANNELS,
    NRC_TOKEN_EMB_DIM,
    HIDDEN_SIZE,
    DROPOUT_RATE,
    WORD_CNN_FILTERS,
    WORD_CNN_KERNELS
)
from empath import Empath
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Set page config
st.set_page_config(
    page_title="Dark Triad Personality Detector",
    page_icon="🔍",
    layout="wide"
)

# Load model and vocabularies
@st.cache_resource
def load_model():
    try:
        # Load vocabularies and other data
        with open("final_enhanced_model_vocab.pkl", 'rb') as f:
            data = pickle.load(f)
            word_vocab = data['word_vocab']
            char_vocab = data['char_vocab']
            nrc_emotions = data['nrc_emotions']
        
        # Create model and load weights
        model = DeepPersonDarkTriad(
            word_vocab_size=len(word_vocab),
            word_emb_dim=EMB_DIM,
            char_vocab_size=len(char_vocab),
            char_emb_dim=CHAR_EMB_DIM,
            char_out_channels=CHAR_OUT_CHANNELS,
            hidden_size=HIDDEN_SIZE,
            nrc_token_feat_dim=len(nrc_emotions),
            nrc_token_emb_dim=NRC_TOKEN_EMB_DIM,
            pretrained_word_emb=None,
            dropout=DROPOUT_RATE
        )
        
        # Load model weights on CPU
        model.load_state_dict(torch.load("final_enhanced_model_model.pth", map_location=torch.device('cpu')))
        model.eval()
        
        # Load NRC lexicon (simplified version for demo)
        nrc_lexicon = defaultdict(lambda: defaultdict(int))
        
        # Initialize Empath
        lexicon = Empath()
        
        return model, word_vocab, char_vocab, nrc_lexicon, nrc_emotions, lexicon
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, None

# Preprocess input text
def preprocess_text(text, word_vocab, char_vocab, nrc_lexicon, nrc_emotions):
    word_ids = torch.tensor(encode_words(text, word_vocab), dtype=torch.long).unsqueeze(0)
    char_ids = torch.tensor(encode_chars(text, char_vocab), dtype=torch.long).unsqueeze(0)
    nrc_token_feats = torch.tensor(
        get_nrc_features_per_token(text, nrc_lexicon, nrc_emotions, MAX_SEQ_LEN), 
        dtype=torch.float32
    ).unsqueeze(0)
    pos_feats = torch.tensor(get_pos_features(text, MAX_SEQ_LEN), dtype=torch.float32).unsqueeze(0)
    readability_feats = torch.tensor(get_readability_features(text), dtype=torch.float32).unsqueeze(0)
    empath_feats = torch.tensor(get_empath_features(text), dtype=torch.float32).unsqueeze(0)
    
    return word_ids, char_ids, nrc_token_feats, pos_feats, readability_feats, empath_feats

# Main app function
def main():
    st.title("🔍 Dark Triad Personality Detector")
    st.markdown("""
    This app analyzes text for signs of Dark Triad personality traits:
    - **Narcissism**: Grandiosity, pride, egotism, and a lack of empathy
    - **Machiavellianism**: Manipulativeness, cunning, deception, and a focus on self-interest
    - **Psychopathy**: Callousness, impulsivity, and antisocial behavior
    """)
    
    # Load model
    model, word_vocab, char_vocab, nrc_lexicon, nrc_emotions, lexicon = load_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure all model files are available.")
        return
    
    # Input options
    input_option = st.radio("Input type:", ("Single text", "Batch upload"))
    
    if input_option == "Single text":
        text = st.text_area("Enter text to analyze:", height=150)
        
        if st.button("Analyze") and text.strip():
            with st.spinner("Analyzing text..."):
                try:
                    # Preprocess
                    inputs = preprocess_text(text, word_vocab, char_vocab, nrc_lexicon, nrc_emotions)
                    
                    # Predict
                    with torch.no_grad():
                        preds, attention_weights = model(*inputs)
                        preds = preds.squeeze(0).numpy()
                    
                    # Display results
                    st.subheader("Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Narcissism", f"{preds[0]*100:.1f}%", 
                                 help="Likelihood of narcissistic traits")
                    
                    with col2:
                        st.metric("Machiavellianism", f"{preds[1]*100:.1f}%", 
                                 help="Likelihood of Machiavellian traits")
                    
                    with col3:
                        st.metric("Psychopathy", f"{preds[2]*100:.1f}%", 
                                 help="Likelihood of psychopathic traits")
                    
                    # Show attention visualization
                    st.subheader("Attention Visualization")
                    tokens = tokenize(clean_text(text))[:MAX_SEQ_LEN]
                    attention_weights = attention_weights.squeeze(0).numpy()[:len(tokens)]
                    
                    # Create a DataFrame for visualization
                    attention_df = pd.DataFrame({
                        'Token': tokens,
                        'Attention': attention_weights[:len(tokens)]
                    })
                    
                    # Show top attended tokens
                    st.bar_chart(attention_df.set_index('Token'))
                    
                    # Show detailed attention
                    st.write("Detailed attention weights:")
                    st.dataframe(attention_df.sort_values('Attention', ascending=False))
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
    
    else:  # Batch upload
        uploaded_file = st.file_uploader("Upload CSV file with text column", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                text_column = st.selectbox("Select text column", df.columns)
                
                if st.button("Analyze Batch"):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, text in enumerate(df[text_column]):
                        status_text.text(f"Processing {i+1}/{len(df)}...")
                        progress_bar.progress((i+1)/len(df))
                        
                        try:
                            inputs = preprocess_text(str(text), word_vocab, char_vocab, nrc_lexicon, nrc_emotions)
                            
                            with torch.no_grad():
                                preds, _ = model(*inputs)
                                preds = preds.squeeze(0).numpy()
                            
                            results.append({
                                'Text': text,
                                'Narcissism': preds[0],
                                'Machiavellianism': preds[1],
                                'Psychopathy': preds[2]
                            })
                        except:
                            results.append({
                                'Text': text,
                                'Narcissism': None,
                                'Machiavellianism': None,
                                'Psychopathy': None
                            })
                    
                    results_df = pd.DataFrame(results)
                    st.subheader("Batch Results")
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results",
                        csv,
                        "dark_triad_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()