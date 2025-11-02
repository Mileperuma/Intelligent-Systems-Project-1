# sentiment/finbert_infer.py
# Purpose: Score each news headline using FinBERT and aggregate results by day.

import pandas as pd
# Import necessary libraries from Hugging Face Transformers and PyTorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np

MODEL_NAME = "ProsusAI/finbert"   # The specific model trained on finance data

def softmax(x):
    """Simple NumPy implementation of the Softmax function."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def score_daily_sentiment(in_csv="sentiment/raw_text.csv", out_csv="sentiment/daily_sentiment.csv"):
    """Load text, score each one, and aggregate results daily."""
    
    df = pd.read_csv(in_csv)
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Load the pre-trained tokenizer and model
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    scores = []
    # Iterate through each headline in the DataFrame
    for _, row in df.iterrows():
        text = str(row['text'])[:512] # Truncate text to model's maximum length
        
        # Tokenize and run inference without gradient tracking (for efficiency)
        inputs = tok(text, return_tensors='pt', truncation=True)
        with torch.no_grad():
            logits = mdl(**inputs).logits.numpy()
            
        # Convert logits to probabilities: [negative, neutral, positive]
        probs = softmax(logits)[0]
        # Save date, positive, negative, and neutral scores
        scores.append((row['date'], probs[2], probs[0], probs[1]))

    # Convert results into a DataFrame
    sdf = pd.DataFrame(scores, columns=['date','pos','neg','neu'])
    
    # Group by date to get daily mean scores and article count
    daily_sent = sdf.groupby('date').agg({
        'pos': 'mean',
        'neg': 'mean',
        'neu': 'mean',
        'date': 'count'
    }).rename(columns={'date': 'count'})

    # Save the final aggregated daily sentiment data
    daily_sent.to_csv(out_csv, index=True)
    print("Saved:", out_csv)

if __name__ == "__main__":
    score_daily_sentiment()