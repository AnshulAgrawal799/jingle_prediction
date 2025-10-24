# Jingle Detection Model
# Detects if jingle is present in audio files using MFCC features and Random Forest classifier

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
import json
import warnings
import time
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


warnings.filterwarnings("ignore")


def extract_features(file_path, sr=22050, n_mfcc=13):
    """Extract MFCC features from an audio file"""
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)  # Average over time frames
    return mfccs


def save_prediction_result(audio_file_path, prediction_result, output_dir='predictions'):
    """
    Save prediction results to a JSON file.

    Args:
        audio_file_path (str): Path to the audio file
        prediction_result (str): Either 'Jingle Present' or 'Jingle Absent'
        output_dir (str): Directory to save prediction results
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Prepare prediction data
        result = {
            'audio_file': os.path.basename(audio_file_path),
            'prediction': prediction_result,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save to JSON file
        output_file = os.path.join(output_dir, 'predictions.json')

        # Load existing data if file exists
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Add new prediction and save
        data.append(result)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(
            f"Saved prediction for {audio_file_path} to {output_file}")
        return True

    except Exception as e:
        logging.error(f"Error saving prediction for {audio_file_path}: {e}")
        return False


def load_data(jingle_dir, non_jingle_dir):
    """Load audio files and extract features and labels"""
    features = []
    labels = []

    # Load jingle files
    for file_name in tqdm(os.listdir(jingle_dir), desc="Loading jingle files"):
        if file_name.endswith(('.wav', '.opus')):
            file_path = os.path.join(jingle_dir, file_name)
            mfccs = extract_features(file_path)
            features.append(mfccs)
            labels.append(1)  # Jingle present

    # Load non-jingle files
    for file_name in tqdm(os.listdir(non_jingle_dir), desc="Loading non-jingle files"):
        if file_name.endswith(('.wav', '.opus')):
            file_path = os.path.join(non_jingle_dir, file_name)
            mfccs = extract_features(file_path)
            features.append(mfccs)
            labels.append(0)  # Jingle absent

    return np.array(features), np.array(labels)


def train_model(jingle_dir='data/train_data/jingle', non_jingle_dir='data/train_data/non_jingle', model_path='jingle_detector_model.pkl', test_size=0.2, random_state=42, n_estimators=100):
    """Train the jingle detector and save the model to model_path."""
    # Load data
    X, y = load_data(jingle_dir, non_jingle_dir)
    if len(X) == 0:
        raise ValueError(
            "No training data found. Ensure WAV files exist in the specified train_data directories.")
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    # Train a Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    # Save the trained model
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    print("Training complete.")


def predict_jingle(file_path, model_path='jingle_detector_model.pkl', save_results=True):
    """
    Predict whether a given audio file contains a jingle.

    Args:
        file_path (str): Path to the audio file
        model_path (str): Path to the trained model
        save_results (bool): Whether to save prediction results to a file

    Returns:
        tuple: (prediction_result, confidence)
    """
    try:
        clf = joblib.load(model_path)
        mfccs = extract_features(file_path)
        mfccs = mfccs.reshape(1, -1)  # Reshape for prediction
        prediction = clf.predict(mfccs)
        proba = clf.predict_proba(mfccs)[0]

        result = "Jingle Present" if prediction[0] == 1 else "Jingle Absent"
        # Get the probability of the predicted class
        confidence = max(proba) * 100

        # Save prediction to file if requested
        if save_results:
            save_prediction_result(file_path, result)

        return result, confidence

    except Exception as e:
        logging.error(f"Error predicting jingle for {file_path}: {e}")
        raise


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    parser = argparse.ArgumentParser(
        description="Train and run a simple jingle detector using MFCC + RandomForest.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_p = subparsers.add_parser("train", help="Train the model")
    train_p.add_argument("--jingle_dir", default="data/train_data/jingle",
                         help="Directory with jingle WAV files")
    train_p.add_argument("--non_jingle_dir", default="data/train_data/non_jingle",
                         help="Directory with non-jingle WAV files")
    train_p.add_argument("--model_path", default="jingle_detector_model.pkl",
                         help="Where to save the trained model")
    train_p.add_argument("--test_size", type=float,
                         default=0.2, help="Test split size (0-1)")
    train_p.add_argument("--n_estimators", type=int,
                         default=100, help="RandomForest n_estimators")
    train_p.add_argument("--random_state", type=int,
                         default=42, help="Random seed")

    # Predict subcommand
    pred_p = subparsers.add_parser(
        "predict", help="Predict on a file or all files in a folder")
    pred_p.add_argument("--input", required=True,
                        help="Path to an audio file or a directory of audio files")
    pred_p.add_argument("--model", default="jingle_detector_model.pkl",
                        help="Path to a trained model file")
    pred_p.add_argument("--output", default="predictions",
                        help="Directory to save prediction results")

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            jingle_dir=args.jingle_dir,
            non_jingle_dir=args.non_jingle_dir,
            model_path=args.model_path,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
    elif args.command == "predict":
        target = args.input

        if not os.path.exists(args.model):
            logging.error(f"Model file not found: {args.model}")
            sys.exit(1)

        if os.path.isdir(target):
            # Process directory of files
            files = [
                os.path.join(target, f) for f in os.listdir(target)
                if os.path.isfile(os.path.join(target, f)) and f.lower().endswith(('.wav', '.opus'))
            ]

            if not files:
                logging.error(f"No WAV or OPUS files found in {target}")
                return

            logging.info(f"Processing {len(files)} files in {target}")
            success_count = 0
            error_count = 0

            for file_path in files:
                try:
                    result, confidence = predict_jingle(
                        file_path,
                        model_path=args.model,
                        save_results=True
                    )
                    print(
                        f"{os.path.basename(file_path)}: {result} (Confidence: {confidence:.1f}%)")
                    success_count += 1
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
                    print(f"Error processing {file_path}: {e}")
                    error_count += 1

            logging.info(
                f"Processing complete. Success: {success_count}, Errors: {error_count}")

        else:
            # Process single file
            if not os.path.exists(target):
                logging.error(f"File not found: {target}")
                sys.exit(1)

            try:
                result, confidence = predict_jingle(
                    target,
                    model_path=args.model,
                    save_results=True
                )
                print(
                    f"Prediction for {os.path.basename(target)}: {result} (Confidence: {confidence:.1f}%)")
                logging.info(
                    f"Prediction saved to {os.path.join(args.output, 'predictions.json')}")
            except Exception as e:
                logging.error(f"Error processing {target}: {e}")
                print(f"Error processing {target}: {e}")
                sys.exit(1)


if __name__ == "__main__":
    main()
