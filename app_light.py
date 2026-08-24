import streamlit as st
from transformers import AutoModelForSequenceClassification, XLMRobertaTokenizer, pipeline

st.title("Review Sentiment")

@st.cache_resource
def load():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

sentiment_model = load()

review = st.text_area("Paste a review")

if st.button("Analyze") and review:
    s = sentiment_model(review)[0]
    st.write("Sentiment:", s["label"], f"({s['score']:.0%})")
