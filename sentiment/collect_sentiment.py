# sentiment/collect_sentiment.py
# Purpose: Standardize raw news data into a clean [date, text] CSV.

import pandas as pd


def save_headlines_csv(input_path, out_csv="sentiment/raw_text.csv"):
    """Prepare raw news data for FinBERT scoring."""
    
    # Load raw data (assumes a CSV/JSON with at least 'date' and 'text' columns)
    df = pd.read_csv(input_path)
    
    # Select only the needed columns and drop any rows with missing data
    df = df[['date','text']].dropna()
    
    # Convert date column to a standardized date-only format
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Save the cleaned text file for the next step (finbert_infer.py)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    # Example command to run the script
    save_headlines_csv("sentiment/raw_text.csv")