"""
This is the script we will use to compute the scores of the participants submission.
Participant don't have the GT_Labels that belong to the test subjects so they
wont be able to run this code. However, we leave the script here for transparency.
"""

import os
import glob
import pandas as pd
from sklearn.metrics import f1_score

# Define patterns for the CSV files in GT_Labels and baselineSubmission.
gt  = '../challengeSubmissions/GT_Labels'
sub = '../challengeSubmissions/baselineSubmission'

gt_pattern = os.path.join(gt, '*', 'Rokoko', 'CSV', '*.csv')
baseline_pattern = os.path.join(sub, '*', 'Rokoko', 'CSV', '*.csv')

# Get sorted lists so that files are paired correctly.
gt_files = sorted(glob.glob(gt_pattern))
baseline_files = sorted(glob.glob(baseline_pattern))

if len(gt_files) != len(baseline_files):
    raise ValueError("The number of ground truth files does not match the number of baseline files.")

all_gt_labels = []
all_pred_labels = []

def extract_labels(file_path):
    """
    Reads the CSV, extracts the second column (label column),
    selects only non-empty cells, discards the first non-empty label,
    then skips any further 'x' labels before converting the rest to integers.
    """
    df = pd.read_csv(file_path)
    
    # Extract the label column (assumed to be the second column)
    labels = df.iloc[:, 1]
    
    # Filter out empty cells (empty string or NaN)
    non_empty = labels[labels.notna() & (labels.astype(str).str.strip() != '')]
    
    # Discard the first non-empty label (which is expected to be 'x')
    non_empty = non_empty.iloc[1:]
    
    # Filter out any remaining 'x' labels (ignoring case and extra spaces)
    filtered = non_empty[non_empty.astype(str).str.strip().str.lower() != 'x']
    
    # Convert the remaining labels to integers (-1, 0, or 1)
    return filtered.astype(int)

# Process each pair of GT and inference files.
for gt_file, baseline_file in zip(gt_files, baseline_files):
    gt_labels = extract_labels(gt_file)
    pred_labels = extract_labels(baseline_file)
    
    # Check that both files have the same number of non-empty labels (after discarding the first one)
    if len(gt_labels) != len(pred_labels):
        raise ValueError(f"Mismatch in number of valid labels between {gt_file} and {baseline_file}.")
    
    # Accumulate labels across files.
    all_gt_labels.extend(gt_labels.tolist())
    all_pred_labels.extend(pred_labels.tolist())

# Compute the aggregated macro F1 score using scikit-learn.
score = f1_score(all_gt_labels, all_pred_labels, average='macro')
print("Macro F1 Score:", score)
