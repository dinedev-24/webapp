import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
import pandas as pd

# Load your model and tokenizer
model = BertForSequenceClassification.from_pretrained('C:/Users/Dine24/Downloads/final_model_backup')  # Provide the correct path
tokenizer = BertTokenizer.from_pretrained('C:/Users/Dine24/Downloads/final_model_backup')  # Provide the correct path

# Define a function to predict sentiment for a given text
def predict_sentiment(text):
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Streamlit app interface
st.title('Sentiment Analysis App')
st.write("This is a simple app that uses a pre-trained model to classify product review sentiment.")

# User input
user_input = st.text_area("Enter a product review:")

if st.button('Predict'):
    if user_input:
        prediction = predict_sentiment(user_input)
        sentiment = "Positive" if prediction == 1 else "Negative" if prediction == 0 else "Neutral"
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a review to predict.")
