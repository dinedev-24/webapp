from transformers import BertForSequenceClassification, BertTokenizer
import torch
import streamlit as st

# Define the path to the model and tokenizer
model_path = 'C:/Users/Dine24/Downloads/final_model_backup'  # Update the path as required

# Load the model and tokenizer using the .safetensors file
model = BertForSequenceClassification.from_pretrained(f'{model_path}/model.safetensors')
tokenizer = BertTokenizer.from_pretrained(f'{model_path}')

# Define the prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

# Streamlit app setup
st.title("Sentiment Analysis App")
text_input = st.text_area("Enter the text for sentiment analysis:")

if text_input:
    sentiment = predict_sentiment(text_input)
    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    st.write(f"The sentiment of the text is: {sentiment_label}")
