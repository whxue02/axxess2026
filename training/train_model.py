import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from fall_detection.feature_engineer import FEATURE_COLS

CSV_PATH   = Path('training/data/keypoints_features.csv')
MODEL_PATH = Path('detection/models/classifier.pkl')

def train():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} frames. Fall rate: {df['label'].mean():.2%}")

    X = df[FEATURE_COLS].values
    y = df['label'].values
    groups = df['video'].values  # used to keep video clips together in splits

    # --- Group split: keep all frames from a video in the same fold ---
    # This prevents data leakage from temporal autocorrelation between frames
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train: {len(X_train)} frames | Test: {len(X_test)} frames")

    # --- Model pipeline ---
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',  # handles class imbalance (fewer falls than non-falls)
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Quick eval
    from sklearn.metrics import classification_report
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['non_fall', 'fall']))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'pipeline': pipeline, 'features': FEATURE_COLS}, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    train()