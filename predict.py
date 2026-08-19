"""
Standalone prediction script for spam detection.

Usage:
    from predict import predict_email
    
    result = predict_email("Congratulations! You won a free prize!")
    print(result)
    
    Output:
        Prediction: SPAM
        Confidence: 97.5%
"""

import json
import re
import torch

from model import SpamDetectionModel, get_device


# File paths
VOCAB_FILE = "vocabulary.json"
MODEL_PATH = "spam_model.pth"


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
    # Load model and move to correct device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)  # Ensure all parameters are on the same device as input
    model.eval()
    return model


def clean_text(text):
    """Clean and normalize text for prediction."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_email(text):
    """
    Predict whether an email is spam or not.
    
    Args:
        text (str): The email message to classify
        
    Returns:
        str: Formatted prediction result with label and confidence
    """
    # Load vocabulary
    print("Loading vocabulary...")
    vocabulary = load_vocabulary(VOCAB_FILE)
    
    # Get device and load model
    print("Loading model...")
    device = get_device()
    model = load_model(vocab_size=len(vocabulary), device=device)
    
    # Clean the text
    cleaned_text = clean_text(text)
    print(f"\nOriginal: {text}")
    print(f"Cleaned:  {cleaned_text}")
    
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
    else:
        prediction_label = "NOT SPAM"
    
    confidence = prob if prob >= 0.5 else 1 - prob
    
    # Format output
    result = f"""Prediction: {prediction_label}
Confidence: {confidence * 100:.1f}%

Details:
- Input tokens: {len(tokens)}
- Model probability of spam: {prob * 100:.2f}%"""
    
    return result


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("SPAM DETECTION - STANALONE PREDICTION")
    print("=" * 60)
    
    test_emails = [
        "Congratulations! You won a free prize!",
        "Hi, can we meet tomorrow at 3pm?",
        "Call now for FREE entry to win £1000!",
        "Meeting at the conference room starting at 2pm"
    ]
    
    for email in test_emails:
        print("\n" + "-" * 60)
        print(f"\nEmail: {email}")
        print("-" * 60)
        result = predict_email(email)
        print(result)
