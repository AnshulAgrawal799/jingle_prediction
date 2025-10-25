# jingle_predict.py
"""
Predict script for Jingle Detector
Run: python scripts/jingle_predict.py --help
Example:
  python scripts/jingle_predict.py --input audio_folder/ --model models/jingle_detector_model.pkl --output predictions
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import librosa
import joblib
import argparse
import os
import sys
import logging
import warnings
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")


def extract_features(file_path, sr=22050, n_mfcc=13):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs


def save_prediction_result(audio_file_path, prediction_result, output_dir='predictions'):
    """Save prediction results to a JSON file (appends to predictions.json)."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        result = {
            'audio_file': os.path.basename(audio_file_path),
            'prediction': prediction_result,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        output_file = os.path.join(output_dir, 'predictions.json')

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

        data.append(result)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(f"Saved prediction for {audio_file_path} to {output_file}")
        return True

    except Exception as e:
        logging.error(f"Error saving prediction for {audio_file_path}: {e}")
        return False


def predict_jingle(file_path, model_path=None, output_dir='predictions', save_results=True):
    """Predict whether a file contains a jingle. Returns (result_text, confidence_percent)."""
    if model_path is None:
        model_path = 'jingle_detector_model.pkl'
    """Predict whether a file contains a jingle. Returns (result_text, confidence_percent)."""
    clf = joblib.load(model_path)
    mfccs = extract_features(file_path)
    mfccs = mfccs.reshape(1, -1)

    prediction = clf.predict(mfccs)
    proba = clf.predict_proba(mfccs)[0] if hasattr(clf, 'predict_proba') else None

    result = "Jingle Present" if int(prediction[0]) == 1 else "Jingle Absent"
    confidence = max(proba) * 100 if proba is not None else None

    if save_results:
        save_prediction_result(file_path, result, output_dir=output_dir)

    return result, confidence


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run jingle prediction on one file or a directory of files")
    parser.add_argument("--input", required=True, help="Path to an audio file or a directory of audio files")
    parser.add_argument("--model", default="jingle_detector_model.pkl", help="Path to trained model file")
    parser.add_argument("--output", default="predictions", help="Directory to save prediction JSON")
    parser.add_argument("--no_save", action='store_true', help="Do not save prediction results to disk")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        logging.error(f"Model file not found: {args.model}")
        sys.exit(1)

    target = args.input

    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in os.listdir(target)
                 if os.path.isfile(os.path.join(target, f)) and f.lower().endswith(('.wav', '.opus'))]

        if not files:
            logging.error(f"No WAV/OPUS files found in {target}")
            sys.exit(1)

        logging.info(f"Processing {len(files)} files in {target}")
        success_count = 0
        error_count = 0

        for file_path in files:
            try:
                result, confidence = predict_jingle(file_path, model_path=args.model, output_dir=args.output, save_results=(not args.no_save))
                conf_str = f" (Confidence: {confidence:.1f}% )" if confidence is not None else ""
                print(f"{os.path.basename(file_path)}: {result}{conf_str}")
                success_count += 1
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                print(f"Error processing {file_path}: {e}")
                error_count += 1

        logging.info(f"Processing complete. Success: {success_count}, Errors: {error_count}")

    else:
        if not os.path.exists(target):
            logging.error(f"File not found: {target}")
            sys.exit(1)

        try:
            result, confidence = predict_jingle(target, model_path=args.model, output_dir=args.output, save_results=(not args.no_save))
            conf_str = f" (Confidence: {confidence:.1f}% )" if confidence is not None else ""
            print(f"Prediction for {os.path.basename(target)}: {result}{conf_str}")
            logging.info(f"Prediction saved to {os.path.join(args.output, 'predictions.json')}")
        except Exception as e:
            logging.error(f"Error processing {target}: {e}")
            print(f"Error processing {target}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
