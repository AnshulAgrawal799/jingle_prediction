# jingle_train.py
"""
Train script for Jingle Detector
Run: python scripts/jingle_train.py --help
Example:
  python scripts/jingle_train.py \
    --jingle_dir data/train_data/jingle \
    --non_jingle_dir data/train_data/non_jingle \
    --model_path models/jingle_detector_model.pkl
"""
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import librosa
import joblib
import argparse
import os
import sys
import logging
import warnings
import time

# Add project root to path (keeps behaviour from original single-file script)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")


def extract_features(file_path, sr=22050, n_mfcc=13):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs


def load_data(jingle_dir, non_jingle_dir):
    """Load audio files and return (X, y) arrays."""
    features = []
    labels = []

    # Jingle files
    for file_name in tqdm(os.listdir(jingle_dir) if os.path.exists(jingle_dir) else [], desc="Loading jingle files"):
        if file_name.lower().endswith((".wav", ".opus")):
            file_path = os.path.join(jingle_dir, file_name)
            mfccs = extract_features(file_path)
            features.append(mfccs)
            labels.append(1)

    # Non-jingle files
    for file_name in tqdm(os.listdir(non_jingle_dir) if os.path.exists(non_jingle_dir) else [], desc="Loading non-jingle files"):
        if file_name.lower().endswith((".wav", ".opus")):
            file_path = os.path.join(non_jingle_dir, file_name)
            mfccs = extract_features(file_path)
            features.append(mfccs)
            labels.append(0)

    return np.array(features), np.array(labels)


def train_model(jingle_dir, non_jingle_dir, model_path, test_size=0.2, random_state=42, n_estimators=100):
    """Train the RandomForest jingle detector and save the model."""
    logging.info("Loading training data...")
    X, y = load_data(jingle_dir, non_jingle_dir)

    if X.size == 0:
        raise ValueError("No training data found. Check your jingle/non_jingle directories.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logging.info("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    logging.info("Training evaluation:")
    logging.info("%s", classification_report(y_test, y_pred))
    logging.info("Accuracy: %.4f", accuracy_score(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    joblib.dump(clf, model_path)
    logging.info(f"Model saved to {model_path}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Train jingle detector (MFCC + RandomForest)")
    parser.add_argument("--jingle_dir", default="data/train_data/jingle", help="Directory with jingle audio files")
    parser.add_argument("--non_jingle_dir", default="data/train_data/non_jingle", help="Directory with non-jingle audio files")
    parser.add_argument("--model_path", default="jingle_detector_model.pkl", help="Where to save the trained model")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--n_estimators", type=int, default=100, help="RandomForest n_estimators")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_model(
        jingle_dir=args.jingle_dir,
        non_jingle_dir=args.non_jingle_dir,
        model_path=args.model_path,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )


if __name__ == "__main__":
    main()
