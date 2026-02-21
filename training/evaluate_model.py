# training/evaluate_model.py
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# Must come before detection imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fall_detection.rf_based.feature_engineer import FEATURE_COLS

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_auc_score,
    classification_report
)
from sklearn.model_selection import GroupShuffleSplit

CSV_PATH   = Path('training/data/keypoints_features.csv')
MODEL_PATH = Path('detection/models/classifier.pkl')

def evaluate():
    df = pd.read_csv(CSV_PATH)
    X = df[FEATURE_COLS].values
    y = df['label'].values
    groups = df['video'].values

    with open(MODEL_PATH, 'rb') as f:
        saved = pickle.load(f)
    pipeline = saved['pipeline']

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(splitter.split(X, y, groups))
    X_test, y_test = X[test_idx], y[test_idx]

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    target_recall = 0.95
    valid = recall[:-1] >= target_recall
    if valid.any():
        best_idx = np.where(valid)[0][np.argmax(precision[:-1][valid])]
        best_threshold = thresholds[best_idx]
        print(f"\nAt recall >= {target_recall}:")
        print(f"  Threshold: {best_threshold:.3f}")
        print(f"  Precision: {precision[best_idx]:.3f}")
        print(f"  Recall:    {recall[best_idx]:.3f}")
    else:
        best_threshold = 0.5
        print("Could not achieve target recall — lowering threshold manually")

    y_pred = (y_proba >= best_threshold).astype(int)
    print("\nClassification report at chosen threshold:")
    print(classification_report(y_test, y_pred, target_names=['non_fall', 'fall']))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(recall, precision, marker='.', label=f'AUC={auc:.3f}')
    axes[0].axvline(target_recall, color='r', linestyle='--', label=f'Target recall={target_recall}')
    axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve'); axes[0].legend()

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['non_fall', 'fall']).plot(ax=axes[1])
    axes[1].set_title(f'Confusion Matrix (threshold={best_threshold:.3f})')

    plt.tight_layout()
    plt.savefig('training/evaluation_results.png', dpi=150)
    print("\nPlot saved to training/evaluation_results.png")

    with open(MODEL_PATH, 'rb') as f:
        saved = pickle.load(f)
    saved['threshold'] = best_threshold
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(saved, f)
    print(f"Threshold {best_threshold:.3f} written back to {MODEL_PATH}")

if __name__ == '__main__':
    evaluate()