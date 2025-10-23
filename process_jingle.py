import os
import numpy as np
import librosa
import joblib
from typing import Union, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_features(file_path: str, sr: int = 22050, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC features from an audio file.
    
    Args:
        file_path: Path to the audio file
        sr: Sample rate (default: 22050)
        n_mfcc: Number of MFCC coefficients to extract (default: 13)
        
    Returns:
        np.ndarray: Extracted MFCC features averaged over time
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)  # Average over time frames
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {str(e)}")
        raise

def predict_jingle(file_path: str, model_path: str = 'jingle_detector_model.pkl') -> Tuple[str, float]:
    """
    Predict if a jingle is present in the audio file.
    
    Args:
        file_path: Path to the audio file to analyze
        model_path: Path to the trained model file (default: 'jingle_detector_model.pkl')
        
    Returns:
        tuple: (prediction_label, confidence_score)
               - prediction_label: "Jingle Present" or "Jingle Absent"
               - confidence_score: Probability score of the prediction [0-1]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load the model
        clf = joblib.load(model_path)
        
        # Extract features
        mfccs = extract_features(file_path)
        mfccs = mfccs.reshape(1, -1)  # Reshape for prediction
        
        # Make prediction
        prediction = clf.predict(mfccs)[0]
        proba = clf.predict_proba(mfccs)[0]
        
        # Get confidence score (probability of the predicted class)
        confidence = proba[prediction]
        
        return ("Jingle Present", float(confidence)) if prediction == 1 else ("Jingle Absent", float(confidence))
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def lambda_handler(event: dict, context: object) -> dict:
    """
    AWS Lambda handler function for jingle detection.
    
    Expected event format:
    {
        'audio_path': 'path/to/audio/file.wav',
        'model_path': 'path/to/model.pkl'  # optional
    }
    """
    try:
        audio_path = event.get('audio_path')
        if not audio_path:
            raise ValueError("Missing required parameter: 'audio_path'")
            
        model_path = event.get('model_path', 'jingle_detector_model.pkl')
        
        # Make prediction
        label, confidence = predict_jingle(audio_path, model_path)
        
        return {
            'statusCode': 200,
            'body': {
                'prediction': label,
                'confidence': confidence,
                'file': audio_path
            }
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'file': event.get('audio_path', 'unknown')
            }
        }

if __name__ == "__main__":
    # Example usage when run directly
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect if a jingle is present in an audio file.')
    parser.add_argument('audio_path', type=str, help='Path to the audio file to analyze')
    parser.add_argument('--model', type=str, default='jingle_detector_model.pkl',
                       help='Path to the trained model (default: jingle_detector_model.pkl)')
    
    args = parser.parse_args()
    
    try:
        label, confidence = predict_jingle(args.audio_path, args.model)
        print(f"Prediction: {label} (Confidence: {confidence:.2f})")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)