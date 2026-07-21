import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK stopwords
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')

load_nltk()

port_stem = PorterStemmer()

# Get absolute path to the directory containing this app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    vec_path = os.path.join(BASE_DIR, 'vectorizer.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

model, vectorizer = load_artifacts()

# Text Preprocessing
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [
        port_stem.stem(word) for word in stemmed_content 
        if word not in stopwords.words('english')
    ]
    return ' '.join(stemmed_content)

# Streamlit UI
st.set_page_config(page_title="Fake News Predictor", page_icon="📰", layout="centered")

st.title("📰 Fake News Prediction Dashboard")
st.write("Enter news headlines or article text below to test the model.")

st.divider()

user_input = st.text_area("News Content / Title:", height=150, placeholder="Paste article text or title here...")

if st.button("Predict", type="primary"):
    if user_input.strip() == "":
        st.warning("Please provide some text to analyze.")
    else:
        # 1. Preprocess
        cleaned_text = stemming(user_input)
        
        # 2. Vectorize
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # 3. Predict
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]
        
        st.subheader("Result")
        if prediction == 1:
            st.error("🚨 **Fake News Detected!**")
            st.metric("Confidence", f"{probability[1]*100:.1f}%")
        else:
            st.success("✅ **Real News Detected!**")
            st.metric("Confidence", f"{probability[0]*100:.1f}%")
