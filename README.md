# Spam Email Detection using PyTorch 📧

A simple, beginner-friendly machine learning project that detects spam emails using LSTM (Long Short-Term Memory) networks in PyTorch.

## 🎯 Project Overview

This project implements a binary text classification model that takes an email/message as input and predicts whether it's SPAM (1) or NOT SPAM (0). The model uses a simple, effective architecture that's perfect for learning neural networks with Python and PyTorch.

**Dataset**: `data/mail_data.csv` - Contains 5,157 emails with labeled categories (ham/spam)

## 📦 Project Structure

```
spam-email-detector/
│
├── data/
│   └── mail_data.csv          # Original dataset
│
├── model.py                   # PyTorch LSTM model definition
├── train.py                   # Training script with preprocessing & evaluation
├── predict.py                 # Standalone prediction function
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── vocabulary.json            # Model vocabulary (auto-generated)
├── spam_model.pth             # Trained model weights (auto-generated)
├── training_curves.png        # Loss visualization (auto-generated)
└── README.md                  # This file
```

## 🛠️ Technologies Used

- **Python 3.9+** - Main programming language
- **PyTorch** - Deep learning framework
- **Pandas, NumPy** - Data manipulation and numerical operations
- **Scikit-learn** - Data splitting and evaluation metrics
- **Matplotlib** - Training curves visualization
- **Streamlit** - Interactive web UI

## 🏗️ Model Architecture

The model follows a simple, beginner-friendly pipeline:

```
Input Text
    ↓
Tokenization (lowercase, remove special chars)
    ↓
Vocabulary Lookup → Integer Indices
    ↓
Padding/Truncation to fixed length (50 tokens)
    ↓
Embedding Layer (64-dimensional vectors)
    ↓
LSTM Layer (captures sequential dependencies)
    ↓
Linear Layer (projects to single output)
    ↓
Sigmoid Activation → Probability [0, 1]
```

### Key Components:

| Component | Description | Parameters |
|-----------|-------------|------------|
| **Embedding** | Converts word indices to dense vectors | Size: 13,596 → 64 |
| **LSTM** | Processes text sequences | Hidden size: 64, Layers: 1 |
| **Linear** | Maps hidden state to prediction | Input: 64 → Output: 1 |

## 📝 How Preprocessing Works

### Text Cleaning:
```python
# Remove punctuation and special characters
text = re.sub(r'[^a-z\s]', ' ', text)
# Normalize whitespace
text = re.sub(r'\s+', ' ', text).strip()
```

### Tokenization:
- Convert to lowercase
- Split on spaces
- Filter out unknown words using `<UNK>` token
- Pad shorter sequences with `<PAD>` token

### Vocabulary Building:
- Extract all unique words from training data
- Assign each word a unique integer ID (2+)
- Special tokens: `<PAD>`=1, `<UNK>`=2

## 🚀 How to Run

### Step 1: Create the Environment

```bash
cd spam-email-detector/
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn torch matplotlib streamlit
```

### Step 2: Train the Model

```bash
python train.py
```

**Output includes:**
- Data loading and cleaning summary
- Vocabulary statistics (number of words, sample mappings)
- Training progress for each epoch
- Test accuracy, precision, recall, F1-score
- Saved model (`spam_model.pth`) and vocabulary (`vocabulary.json`)

### Step 3: Run Standalone Predictions

```bash
python predict.py
```

Test the model with example emails in your terminal.

### Step 4: Start Streamlit Web App

```bash
streamlit run app.py
```

Opens interactive web interface at `http://localhost:8501`

## 📊 Evaluation Metrics

The model reports:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions (ham + spam) / total |
| **Precision** | True spam / (True spam + False positives) |
| **Recall** | True spam / (True spam + False negatives) |
| **F1-Score** | Harmonic mean of precision and recall |

### Example Output:
```
Test Accuracy: 87.60%
Precision: 89.45%
Recall: 87.23%
F1-Score: 88.33%
```

## 🔧 Model Parameters (Configurable in `train.py`)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `embedding_dim` | 64 | Dimension of word embeddings |
| `hidden_dim` | 64 | LSTM hidden state size |
| `num_layers` | 1 | Number of LSTM layers |
| `dropout` | 0.2 | Dropout probability for regularization |
| `batch_size` | 32 | Samples per training batch |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `epochs` | 5 | Number of training epochs |
| `max_seq_length` | 50 | Maximum sequence length |

## 📖 Usage Examples

### Using the Prediction Function:

```python
from predict import predict_email

result = predict_email("Congratulations! You won a free prize!")
print(result)

# Output:
# Prediction: SPAM
# Confidence: 97.5%
# Details:
# - Input tokens: 6
# - Model probability of spam: 97.48%
```

### Using the Web Interface:

1. Open `http://localhost:8501` in your browser
2. Paste an email message in the text area
3. Click **Predict** to see the classification result

## 🎓 Learning Points

This project demonstrates key deep learning concepts:

1. **Sequence Processing**: How LSTMs handle text as sequential data
2. **Embeddings**: Converting discrete tokens to continuous vectors
3. **Train/Validate/Test Splits**: Proper data partitioning for evaluation
4. **Class Imbalance**: Handling datasets with unequal class distribution
5. **Hyperparameter Tuning**: Understanding the impact of model architecture choices
6. **Full Pipeline**: From raw text to prediction in a production-ready system

## 📝 Dataset Information

The dataset (`data/mail_data.csv`) contains:

- **Total records**: 5,572 emails
- **After cleaning**: 5,157 unique emails (415 duplicates removed)
- **Class distribution**:
  - NOT SPAM (ham): 4,516 messages (87.6%)
  - SPAM: 641 messages (12.4%)

The data is formatted as:
```csv
Category,Message
ham,"Hi, can we meet tomorrow at 3pm?"
spam,"Free entry in 2 a wkly comp to win FA Cup final..."
```

## ⚠️ Limitations

- Model was trained on only 5 epochs - may benefit from more training
- Simple architecture (no attention, no transformers) - not state-of-the-art
- Class imbalance may require oversampling techniques for production use
- Test accuracy of ~88% is good but not perfect

## 📚 Next Steps

To improve this model:

1. **More epochs**: Increase from 5 to 20+ training epochs
2. **Better hyperparameters**: Tune learning rate, batch size, hidden dimensions
3. **Advanced preprocessing**: Add stemming, lemmatization, n-grams
4. **Multi-layer LSTM**: Stack multiple LSTM layers for deeper features
5. **Ensemble methods**: Combine predictions from multiple models

## 📄 License

This project is for educational purposes. Free to use and modify!

---

**Built with ❤️ using PyTorch and Streamlit**
