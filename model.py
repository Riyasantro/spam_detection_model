"""
Spam Email Detection Model using PyTorch LSTM

This file defines the neural network architecture for spam detection.
Model: Text -> Tokenization -> Embedding -> LSTM -> Linear -> Sigmoid (Binary Classification)
"""

import torch
import torch.nn as nn


class SpamDetectionModel(nn.Module):
    """
    LSTM-based model for binary spam classification.
    
    Architecture:
    1. Embedding Layer: Converts word indices to dense vectors
    2. LSTM Layer: Captures sequential dependencies in text
    3. Linear Layer: Projects hidden state to single output
    4. Sigmoid: Outputs probability (0-1) for spam detection
    
    Parameters:
        embedding_dim (int): Size of word embeddings (default: 64)
        hidden_dim (int): LSTM hidden state size (default: 64)
        num_layers (int): Number of LSTM layers (default: 1)
        dropout (float): Dropout probability for regularization (default: 0.2)
        vocab_size (int): Vocabulary size (number of unique words) (default: 10000)
    """
    
    def __init__(self, embedding_dim=64, hidden_dim=64, num_layers=1, dropout=0.2, vocab_size=10000):
        super(SpamDetectionModel, self).__init__()
        
        # Embedding layer: word index -> dense vector
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        # LSTM layer: processes sequences and captures context
        # batch_first=True: input shape is (batch_size, seq_len)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Linear layer: LSTM output -> single neuron for binary classification
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        print(f"Model initialized:")
        print(f"  - Embedding dimension: {embedding_dim}")
        print(f"  - Hidden dimension: {hidden_dim}")
        print(f"  - LSTM layers: {num_layers}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Vocabulary size: {vocab_size}")
    
    def forward(self, text_input):
        """
        Forward pass through the network.
        
        Args:
            text_input (Tensor): Shape (batch_size, seq_len) containing word indices
            
        Returns:
            Tensor: Shape (batch_size, 1) with spam probabilities
        """
        # Ensure input is on same device as model parameters
        if self.embedding.weight.device != text_input.device:
            text_input = text_input.to(self.embedding.weight.device)
        
        # Step 1: Embedding layer converts word indices to dense vectors
        embedded = self.embedding(text_input)
        
        # Step 2: Apply dropout to embeddings
        embedded = self.dropout(embedded)
        
        # Step 3: LSTM processes the sequence
        lstm_output, _ = self.lstm(embedded)
        
        # Get the final hidden state: (batch_size, 1, hidden_dim)
        lstm_output = lstm_output[:, -1, :]
        
        # Step 4: Dropout on LSTM output
        lstm_output = self.dropout(lstm_output)
        
        # Step 5: Linear layer maps hidden state to single output
        output = self.fc(lstm_output)
        
        return output


def get_device():
    """
    Automatically select GPU or CPU.
    
    Returns:
        torch.device: The device to use for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
