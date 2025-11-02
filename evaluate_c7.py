# evaluate_c7.py
# Purpose: Load saved metrics from train_c7.py and generate Confusion Matrix plots for comparison.

import json, os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "plots_c7" # Directory where the metrics and plots are stored

def load_metrics(tag):
    """Load the JSON file containing the trained model's performance metrics."""
    with open(os.path.join(OUT_DIR, f"c7_metrics_{tag}.json"), "r") as f:
        return json.load(f)

def plot_cm(cm, title, fname):
    """
    Generate and save a visualization of the Confusion Matrix.
    
    Args:
        cm (list/array): The 2x2 confusion matrix data.
        title (str): Title for the plot.
        fname (str): Filename to save the plot as.
    """
    cm = np.array(cm)
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest') # Display the matrix as an image
    
    # Annotate each cell with the corresponding count (True Pos, False Neg, etc.)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha='center', va='center')
            
    # Set the axis labels (0=Down, 1=Up)
    plt.xticks([0,1], ['Down','Up']); plt.yticks([0,1], ['Down','Up'])
    plt.title(title); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    
    # Save the figure and close it
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150); plt.close(fig)

if __name__ == "__main__":
    # Load metrics for the two runs
    base = load_metrics("baseline_no_sent")
    full = load_metrics("with_sent")

    # Print the raw metrics to the console
    print("Baseline:", base)
    print("With Sentiment:", full)

    # Generate and save the Confusion Matrix plots
    plot_cm(base["confusion_matrix"], "Baseline Confusion", "cm_baseline.png")
    plot_cm(full["confusion_matrix"], "Sentiment Confusion", "cm_with_sentiment.png")