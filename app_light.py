"""
Tourist Review Sentiment — ultra-light Streamlit demo (single model).

Just sentiment analysis, sized to run comfortably on Streamlit Community Cloud's
free tier (1GB RAM). For the full app (sentiment + topic + summary), see app.py
and run it locally.

Run with: streamlit run app_light.py
"""

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Tourist Review Sentiment", page_icon="📝")


@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


sentiment_model = load_model()

st.title("📝 Tourist Review Sentiment")
st.caption("Bahrain restaurant & hotel reviews — pretrained Hugging Face model, no training required.")

review = st.text_area("Paste a review:", height=150)

if st.button("Analyze") and review.strip():
    result = sentiment_model(review)[0]
    output = {
        "label": result["label"],
        "score": result["score"],
        "metadata": "huggingface_AI_model",
    }

    st.metric("Sentiment", output["label"], f"{output['score']:.0%} confidence")

    with st.expander("Raw model output"):
        st.json(output)
