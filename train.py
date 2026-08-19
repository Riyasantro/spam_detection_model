"""
Training script for Spam Detection model using LSTM.

This script:
1. Loads and preprocesses the dataset
2. Creates train/test splits  
3. Builds vocabulary from training data
4. Trains the LSTM model with validation
5. Evaluates metrics and saves the model
"""

import os
import re
import json
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt

from model import SpamDetectionModel, get_device
import torch.nn as nn


# ============== Configuration ==============
class Config:
    """Training hyperparameters and paths."""
    DATA_PATH = "data/mail_data.csv"
    VOCAB_FILE = "vocabulary.json"
    MODEL_PATH = "spam_model.pth"
    
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    DROPOUT = 0.2
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 5
    
    MAX_SEQ_LENGTH = 50
    OOV_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    
    RANDOM_SEED = 42


# ============== Data Loading ==============

def load_and_clean_data(filepath):
    """
    Load CSV and clean the data.
    
    The dataset is comma-separated: Category,Message (with header row)
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataframe with 'label' (0/1) and 'message' columns
    """
    # Read the CSV file
    df = pd.read_csv(filepath, encoding='latin-1')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head(3)}")
    
    # Process based on column names found
    if 'Category' in df.columns or 'category' in df.columns:
        label_col = 'Category' if 'Category' in df.columns else [c for c in df.columns][0]
        print(f"Using column '{label_col}' as label column")
        
        # Create new dataframe with proper structure
        df_new = pd.DataFrame()
        df_new['label'] = df[label_col].apply(lambda x: 1 if str(x).strip().lower() == 'spam' else 0)
        
        # Find message column
        msg_col_candidates = [c for c in df.columns if c.lower() in ['message', 'msg', 'text']]
        if msg_col_candidates:
            msg_col = msg_col_candidates[0]
            df_new['message'] = df[msg_col]
        else:
            df_new['message'] = ''
        
        df = df_new
        print(f"  Reshaped to columns: {df.columns.tolist()}")

    elif 'label' not in df.columns and len(df.columns) >= 2:
        # Handle alternative column structure
        label_col = [c for c in df.columns if c.lower() in ['category', 'spam', 'ham']][0]
        
        df_new = pd.DataFrame()
        df_new['label'] = df[label_col].apply(lambda x: 1 if str(x).strip().lower() == 'spam' else 0)
        
        msg_col_candidates = [c for c in df.columns if c.lower() in ['message', 'msg', 'text']]
        if msg_col_candidates:
            msg_col = msg_col_candidates[0]
            df_new['message'] = df[msg_col]
        else:
            df_new['message'] = ''
        
        df = df_new
        print(f"  Reshaped to columns: {df.columns.tolist()}")

    # If we still don't have proper columns, try simpler approach
    if 'label' not in df.columns or len(df.columns) < 2:
        print("  Using simplified column detection...")
        if len(df.columns) >= 2:
            label_col = df.columns[0]
            msg_col = df.columns[1]
            df = pd.DataFrame()
            df['label'] = df[label_col].apply(lambda x: 1 if str(x).strip().lower() == 'spam' else 0)
            df['message'] = df[msg_col]

    print(f"After preprocessing shape: {df.shape}")
    
    # Remove rows with missing values
    initial_count = len(df)
    df = df.dropna(subset=['label', 'message'])
    print(f"After removing missing values: {len(df)} rows (removed {initial_count - len(df)})")
    
    # Remove duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['message', 'label'])
    print(f"After removing duplicates: {len(df)} rows (removed {before_dedup - len(df)})")
    
    # Show class distribution
    total = len(df)
    spam_count = (df['label'] == 1).sum()
    ham_count = (df['label'] == 0).sum()
    
    print(f"\nClass Distribution:")
    print(f"  - NOT SPAM (ham): {ham_count} ({100*ham_count/total:.1f}%)")
    print(f"  - SPAM: {spam_count} ({100*spam_count/total:.1f}%)")
    
    return df


# ============== Text Preprocessing ==============

def clean_text(text):
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_vocabulary(df, min_freq=1):
    """Build vocabulary from training data."""
    all_messages = df['message'].dropna().tolist()
    
    word_freq = {}
    for message in all_messages:
        words = message.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    vocabulary = {Config.OOV_TOKEN: 0, Config.PAD_TOKEN: 1}
    
    idx = 2
    for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
        if freq >= min_freq and word not in [Config.OOV_TOKEN, Config.PAD_TOKEN]:
            vocabulary[word] = idx
            idx += 1
    
    vocab_size = len(vocabulary)
    
    print(f"\nVocabulary created with {vocab_size} words")
    print(f"Sample vocabulary (first 10):")
    for i, (word, idx) in enumerate(list(vocabulary.items())[:10]):
        print(f"  '{word}' -> {idx}")
    
    return vocabulary, vocab_size


def encode_tokens(tokens, vocabulary):
    """Convert tokens to integer indices."""
    encoded = []
    for token in tokens:
        if token in vocabulary:
            encoded.append(vocabulary[token])
        else:
            encoded.append(vocabulary[Config.OOV_TOKEN])
    return encoded


def pad_sequence(sequence, max_length, vocab):
    """Pad or truncate sequence to fixed length."""
    if len(sequence) >= max_length:
        return sequence[:max_length]
    else:
        return sequence + [vocab[Config.PAD_TOKEN]] * (max_length - len(sequence))


# ============== Dataset and DataLoader ==============

class SpamDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for spam detection."""
    
    def __init__(self, texts, labels, vocabulary, max_length):
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        cleaned = clean_text(self.texts[idx])
        tokens = cleaned.split()
        encoded = encode_tokens(tokens, self.vocabulary)
        padded = pad_sequence(encoded, self.max_length, self.vocabulary)
        
        return torch.tensor(padded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


# ============== Training Functions ==============

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_text, batch_labels in dataloader:
        batch_text = batch_text.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_text)
        loss = criterion(predictions, batch_labels.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_text, batch_labels in dataloader:
            batch_text = batch_text.to(device)
            batch_labels = batch_labels.to(device)
            
            predictions = model(batch_text)
            loss = criterion(predictions, batch_labels.unsqueeze(1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def evaluate_metrics(model, dataloader, device):
    """Calculate classification metrics."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_text, batch_labels in dataloader:
            batch_text = batch_text.to(device)
            
            outputs = model(batch_text).squeeze(1)
            predictions = (outputs >= 0.5).long()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    return {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, zero_division=0),
        'f1': f1_score(all_labels, all_predictions, zero_division=0)
    }


# ============== Visualization ==============

def plot_training_curves(train_losses, val_losses, epochs):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='s')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: training_curves.png")


# ============== Main Training Function ==============

def train_model():
    """Main training function."""
    
    device = get_device()
    
    print("=" * 60)
    print("SPAM DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load and clean data
    print("\n[1/7] Loading and cleaning data...")
    df = load_and_clean_data(Config.DATA_PATH)
    
    # Step 2: Split data
    print("\n[2/7] Splitting data (80% train, 20% test)...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=Config.RANDOM_SEED,
        stratify=df['label']
    )
    
    print(f"  - Training samples: {len(train_df)}")
    print(f"  - Testing samples: {len(test_df)}")
    
    # Step 3: Clean text and build vocabulary (only from training data)
    print("\n[3/7] Cleaning text and building vocabulary...")
    train_df['cleaned'] = train_df['message'].apply(clean_text)
    test_df['cleaned'] = test_df['message'].apply(clean_text)  # Also clean test set
    
    vocabulary, vocab_size = create_vocabulary(train_df)
    
    # Step 4: Create Dataset and DataLoader
    print("\n[4/7] Creating Dataset and DataLoader...")
    
    train_dataset = SpamDataset(
        texts=train_df['cleaned'].tolist(),
        labels=train_df['label'].tolist(),
        vocabulary=vocabulary,
        max_length=Config.MAX_SEQ_LENGTH
    )
    
    test_dataset = SpamDataset(
        texts=test_df['cleaned'].tolist(),
        labels=test_df['label'].tolist(),
        vocabulary=vocabulary,
        max_length=Config.MAX_SEQ_LENGTH
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Testing batches: {len(test_loader)}")
    
    # Step 5: Initialize model
    print("\n[5/7] Initializing model...")
    model = SpamDetectionModel(
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        vocab_size=vocab_size
    )
    
    model.to(device)
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.5]))  # Weight positive class higher due to imbalance
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    print(f"  - Optimizer: Adam (lr={Config.LEARNING_RATE})")
    print(f"  - Loss function: BCEWithLogitsLoss (with class weighting for imbalanced data)")
    
    # Step 6: Training loop
    print("\n[6/7] Training model...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        val_loss = validate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        
        metrics = evaluate_metrics(model, test_loader, device)
        
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Test Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall: {metrics['recall']*100:.2f}%")
        print(f"  F1-Score: {metrics['f1']*100:.2f}%")
    
    # Step 7: Save model and vocabulary
    print("\n[7/7] Saving model and vocabulary...")
    
    with open(Config.VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f)
    print(f"  Vocabulary saved to: {Config.VOCAB_FILE}")
    
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"  Model saved to: {Config.MODEL_PATH}")
    
    plot_training_curves(train_losses, val_losses, Config.EPOCHS)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
