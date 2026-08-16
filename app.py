"""
Tourist Review Analyzer — minimal Streamlit app.

Paste a restaurant/hotel review and get:
  - sentiment (positive/negative)
  - main topic (zero-shot)
  - a short summary (only shown for longer reviews)

Run with: streamlit run app.py
"""

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Tourist Review Analyzer", page_icon="📝")


@st.cache_resource
def load_models():
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return sentiment, zero_shot, summarizer


sentiment_model, zero_shot_model, summarizer = load_models()

CANDIDATE_TOPICS = [
    "Food Quality", "Service", "Price", "Cleanliness",
    "Atmosphere", "Location", "Waiting Time",
]

st.title("📝 Tourist Review Analyzer")
st.caption("Bahrain restaurant & hotel reviews — pretrained Hugging Face models, no training required.")

review = st.text_area("Paste a review:", height=150)

if st.button("Analyze") and review.strip():
    sentiment_result = sentiment_model(review)[0]
    sentiment_output = {
        "label": sentiment_result["label"],
        "score": sentiment_result["score"],
        "metadata": "huggingface_AI_model",
    }

    topic_result = zero_shot_model(review, CANDIDATE_TOPICS)
    topic_output = {
        "label": topic_result["labels"][0],
        "score": topic_result["scores"][0],
        "metadata": "huggingface_AI_model",
    }

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sentiment", sentiment_output["label"], f"{sentiment_output['score']:.0%} confidence")
    with col2:
        st.metric("Main topic", topic_output["label"], f"{topic_output['score']:.0%} confidence")

    if len(review.split()) > 40:
        summary = summarizer(review, max_length=40, min_length=10, do_sample=False)[0]["summary_text"]
        st.subheader("Summary")
        st.write(summary)

    with st.expander("Raw model output"):
        st.json({"sentiment": sentiment_output, "topic": topic_output})
