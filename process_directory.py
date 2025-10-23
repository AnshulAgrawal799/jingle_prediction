#!/usr/bin/env python3
"""
Process all audio files in a directory using the jingle detection model.

Usage:
    python process_directory.py input_directory [--model MODEL_PATH] [--output OUTPUT_CSV]

Example:
    python process_directory.py audio_samples/ --model jingle_detector_model.pkl --output results.csv
"""

import os
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any
from process_jingle import predict_jingle

def get_audio_files(directory: str) -> List[str]:
    """Get a list of audio files in the specified directory."""
    audio_extensions = {'.wav', '.mp3', '.ogg', '.opus', '.flac', '.aac', '.m4a'}
    return [
        str(f) for f in Path(directory).glob('*')
        if f.suffix.lower() in audio_extensions and f.is_file()
    ]

def process_audio_files(directory: str, model_path: str) -> List[Dict[str, Any]]:
    """Process all audio files in the directory and return results."""
    audio_files = get_audio_files(directory)
    if not audio_files:
        print(f"No audio files found in {directory}")
        return []

    results = []
    for file_path in audio_files:
        try:
            prediction, confidence = predict_jingle(file_path, model_path)
            results.append({
                'file': os.path.basename(file_path),
                'prediction': prediction,
                'confidence': f"{confidence:.4f}",
                'status': 'success'
            })
            print(f"Processed {os.path.basename(file_path)}: {prediction} ({confidence:.2f}%)")
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            results.append({
                'file': os.path.basename(file_path),
                'prediction': 'error',
                'confidence': '0',
                'status': f"error: {str(e)}"
            })
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save results to a CSV file."""
    if not results:
        print("No results to save.")
        return
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Process audio files in a directory for jingle detection.')
    parser.add_argument('input_dir', help='Directory containing audio files to process')
    parser.add_argument('--model', default='jingle_detector_model.pkl',
                      help='Path to the trained model (default: jingle_detector_model.pkl)')
    parser.add_argument('--output', default='jingle_detection_results.csv',
                      help='Output CSV file path (default: jingle_detection_results.csv)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}")
        return
    
    if not os.path.isfile(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    print(f"Processing audio files in: {args.input_dir}")
    print(f"Using model: {args.model}")
    
    results = process_audio_files(args.input_dir, args.model)
    
    if results:
        save_results(results, args.output)
    else:
        print("No files were processed.")

if __name__ == "__main__":
    main()
