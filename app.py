"""
Streamlit Web Application for Spam Email Detection

A simple, user-friendly interface to test the spam detection model.

Features:
- Text input area for entering email messages
- One-click prediction
- Confidence score display
- Clear and intuitive UI
"""

import streamlit as st
import json
import re
import torch

from model import SpamDetectionModel, get_device


# Page configuration
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)


def load_vocabulary(filepath):
    """Load vocabulary from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model(vocab_size, device):
    """Load trained model weights."""
    model = SpamDetectionModel(
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.2,
        vocab_size=vocab_size
    )
    model.load_state_dict(torch.load("spam_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


def clean_text(text):
    """Clean and normalize text for prediction."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_spam(text, model, device, vocabulary):
    """Predict whether an email is spam or not."""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize
    tokens = cleaned_text.split()
    
    # Encode to integers
    encoded = []
    for token in tokens:
        if token in vocabulary:
            encoded.append(vocabulary[token])
        else:
            encoded.append(vocabulary.get("<UNK>", 2))
    
    # Pad/truncate to max length (50)
    max_length = 50
    if len(encoded) >= max_length:
        encoded = encoded[:max_length]
    else:
        encoded = encoded + [vocabulary.get("<PAD>", 1)] * (max_length - len(encoded))
    
    # Create tensor and predict
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(input_tensor).squeeze()
        prob = torch.sigmoid(output).item()
    
    # Convert probability to prediction (threshold at 0.5)
    if prob >= 0.5:
        prediction_label = "SPAM"
        color = "#ff4b4b"  # Red
    else:
        prediction_label = "NOT SPAM"
        color = "#4caf50"  # Green
    
    confidence = prob if prob >= 0.5 else 1 - prob
    
    return {
        "prediction": prediction_label,
        "confidence": confidence,
        "color": color,
        "probability": prob
    }


# Title
st.title("📧 Spam Email Detection")

st.markdown("""
## Simple LSTM-based spam detection using PyTorch

Enter an email message below and click **Predict** to see if it's classified as SPAM or NOT SPAM.
""")

# Text input area
text_input = st.text_area(
    "Paste your email message here:",
    height=150,
    placeholder="Example: Congratulations! You won a free prize! Click here now!"
)

# Predict button
if st.button("Predict", type="primary"):
    if text_input.strip():
        try:
            # Load model (cached for efficiency)
            if "model" not in st.session_state:
                st.info("Loading model...")
                device = get_device()
                vocabulary = load_vocabulary("vocabulary.json")
                st.session_state["model"] = load_model(vocab_size=len(vocabulary), device=device)
                st.session_state["device"] = device
                st.session_state["vocabulary"] = vocabulary
            
            model = st.session_state["model"]
            device = st.session_state["device"]
            vocabulary = st.session_state["vocabulary"]
            
            # Make prediction
            result = predict_spam(text_input, model, device, vocabulary)
            
            # Display results
            st.markdown("### Result")
            
            # Create a styled prediction box
            st.markdown(f"""
            <div style="background-color: {result['color']}; color: white; 
                        padding: 20px; border-radius: 10px; text-align: center; 
                        font-size: 24px; font-weight: bold;">
                Prediction: {result['prediction']}
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence score
            st.markdown(f"""
            <div style="text-align: center; margin-top: 20px;">
                <p style="font-size: 18px; font-weight: bold;">Confidence: 
                {result['confidence'] * 100:.1f}%</p>
                <p style="color: gray;">Model probability of spam: {result['probability'] * 100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("⚠️ Please enter some text to predict.")


# Add a section with example emails
st.markdown("### 📝 Example Messages")

example_emails = [
    "Congratulations! You won a free prize!",
    "Hi, can we meet tomorrow at 3pm?",
    "Call now for FREE entry to win £1000!",
    "Meeting at the conference room starting at 2pm",
    "URGENT: Your account has been suspended. Click here!"
]

with st.expander("Try example emails"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Example 1 (Spam)", key="ex1"):
            st.session_state["example_text"] = "Congratulations! You won a free prize!"
    
    with col2:
        if st.button("Example 2 (Not Spam)", key="ex2"):
            st.session_state["example_text"] = "Hi, can we meet tomorrow at 3pm?"
    
    with col3:
        if st.button("Example 3 (Spam)", key="ex3"):
            st.session_state["example_text"] = "Call now for FREE entry to win £1000!"


if "example_text" in st.session_state and text_input == st.session_state["example_text"]:
    # Auto-predict if example was clicked
    try:
        device = get_device()
        vocabulary = load_vocabulary("vocabulary.json")
        model = load_model(vocab_size=len(vocabulary), device=device)
        
        result = predict_spam(st.session_state["example_text"], model, device, vocabulary)
        
        st.markdown(f"Prediction: {result['prediction']} (Confidence: {result['confidence']*100:.1f}%)")
    except Exception as e:
        st.error(f"Error: {str(e)}")


# Model information footer
st.markdown("---")
st.markdown("""
### ℹ️ About This Model

- **Architecture**: LSTM (Long Short-Term Memory) Network
- **Framework**: PyTorch with GPU/CPU auto-detection
- **Training Data**: 5,157 emails (4,516 ham, 641 spam)
- **Performance**: ~88% test accuracy
""")
