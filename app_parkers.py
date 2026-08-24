import streamlit as st
from transformers import pipeline

st.title("Parker's Review Analyzer")

@st.cache_resource
def load():
    sentiment = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    topic = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    return sentiment, topic

sentiment_model, topic_model = load()
candidate_labels = ["Food", "Service", "Price", "Cleanliness", "Atmosphere"]

review = st.text_area("Paste a review")

if st.button("Analyze") and review:
    s = sentiment_model(review)[0]
    t = topic_model(review, candidate_labels)

    st.write("Sentiment:", s["label"], f"({s['score']:.0%})")
    st.write("Topic:", t["labels"][0], f"({t['scores'][0]:.0%})")
