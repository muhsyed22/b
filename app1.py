# app.py - STREAMLIT CLOUD COMPATIBLE VERSION

import os
import pickle
import numpy as np
import streamlit as st
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import requests

# ------------------- DOWNLOAD NLTK DATA -------------------
# This is CRITICAL for Streamlit Cloud deployment
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# ------------------- CONFIG -------------------
MODEL_PATH = "word2vec.model"
VECS_PATH = "vectors.npy"
PKL_PATH = "sampled_products.pkl"
GOOGLE_DRIVE_FILE_ID = "1gXAC70uJLSC5pzccsxxy2MiPwP5VOv5N"

# ------------------- DOWNLOAD MODEL FROM GOOGLE DRIVE -------------------
@st.cache_resource
def download_model_from_gdrive():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading Word2Vec model from Google Drive (one-time setup)...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}&export=download&confirm=t"
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

# ------------------- LOAD ARTIFACTS -------------------
@st.cache_resource
def load_artifacts():
    # Download model if missing
    download_model_from_gdrive()
    
    # 1. Model
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    model = Word2Vec.load(MODEL_PATH)
    
    # 2. Vectors
    if not os.path.exists(VECS_PATH):
        st.error(f"Vectors not found: {VECS_PATH}. Please upload '{VECS_PATH}' to your repo.")
        st.stop()
    vectors = np.load(VECS_PATH)
    
    # 3. Product list
    if not os.path.exists(PKL_PATH):
        st.error(f"Sampled products not found: {PKL_PATH}. Please upload '{PKL_PATH}' to your repo.")
        st.stop()
    with open(PKL_PATH, "rb") as f:
        products = pickle.load(f)
    
    return model, vectors, products

model, vectors, products = load_artifacts()

# ------------------- HELPERS -------------------
def vectorize_query(query: str) -> np.ndarray:
    """Turn a free-text query into a 100-dim vector"""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    toks = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(query)
            if w.isalpha() and w.lower() not in stop_words]
    
    if not toks:
        return np.zeros(100)
    
    vecs = [model.wv[t] for t in toks if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

# ------------------- UI -------------------
st.title("🛍️ Clothing Product Search (Word2Vec)")
st.markdown("Search for clothing products using natural language queries!")

query = st.text_input("Enter a product description or keywords", "red summer dress with pockets")

if query:
    q_vec = vectorize_query(query).reshape(1, -1)
    sims = cosine_similarity(q_vec, vectors).flatten()
    top_k = 5
    top_idx = np.argsort(sims)[-top_k:][::-1]
    
    st.write(f"**Top {top_k} matches** (cosine similarity):")
    
    for rank, idx in enumerate(top_idx, 1):
        prod = products[idx]
        sim = sims[idx]
        
        with st.expander(f"#{rank} – {prod.get('title', 'Untitled')} (similarity={sim:.3f})"):
            st.write("**Title:**", prod.get("title"))
            st.write("**Description:**", prod.get("description"))
            st.write("**Features:**", prod.get("feature"))
            if prod.get("imageURL"):
                st.image(prod["imageURL"], width=200)
