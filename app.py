import streamlit as st
from transformers import pipeline
import re

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬", layout="centered")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Built with `distilbert-base-uncased-finetuned-sst-2-english` · 89% accuracy on IMDB")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def clean_text(text):
    text = re.sub('<.*?>', '', text)
    text = text.strip()
    return text

classifier = load_model()

st.markdown("### Enter a movie review")
user_input = st.text_area("", placeholder="e.g. This movie was absolutely brilliant...", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        result = classifier(cleaned, truncation=True, max_length=512)[0]
        label = result["label"]
        score = round(result["score"] * 100, 2)

        if label == "POSITIVE":
            st.success(f"✅ Positive — {score}% confidence")
        else:
            st.error(f"❌ Negative — {score}% confidence")

        with st.expander("See details"):
            st.write(f"**Cleaned input:** {cleaned[:300]}...")
            st.write(f"**Label:** {label}")
            st.write(f"**Confidence:** {score}%")
    else:
        st.warning("Please enter a review first.")

st.markdown("---")
st.caption("Model: DistilBERT · Dataset: IMDB 50K reviews · Accuracy: 89%")