# Jingle Detection System

A machine learning-based system for detecting the presence of jingles in audio files. The system uses MFCC (Mel-frequency cepstral coefficients) features and a Random Forest classifier to identify jingles in audio streams.

## Features

- 🎵 Supports multiple audio formats: WAV, MP3, OPUS, OGG, FLAC, AAC, M4A
- ⚡ Fast and efficient audio processing
- 📁 Batch processing of multiple files
- 📊 Detailed results output to CSV
- 🖥️ Easy-to-use command line interface
- 🐍 Python 3.7+ compatible

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd jingle_prediction
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note for Windows users**: If you encounter issues, try:
   > ```bash
   > pip install numpy --only-binary=:all:
   > pip install -r requirements.txt
   > ```

## Usage

### Single File Processing

To process a single audio file:

```bash
python process_jingle.py path/to/audio.wav --model path/to/model.pkl
```

**Output Example:**
```
Prediction for audio.wav: Jingle Present (0.92)
```

### Batch Processing

To process all audio files in a directory:

```bash
python process_directory.py path/to/audio_files --model jingle_detector_model.pkl --output results.csv
```

**Arguments:**
- `input_dir`: Directory containing audio files (required)
- `--model`: Path to the trained model (default: 'jingle_detector_model.pkl')
- `--output`: Output CSV file path (default: 'jingle_detection_results.csv')

**Example Output CSV:**
```csv
file,prediction,confidence,status
sample1.opus,Jingle Present,0.9231,success
sample2.wav,Jingle Absent,0.8512,success
error_file.opus,error,0,error: Invalid audio file
```

## Project Structure

```
jingle_prediction/
├── process_jingle.py      # Main script for single file processing
├── process_directory.py   # Script for batch processing
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── jingle_detector_model.pkl  # Pre-trained model (if included)
```

## Model Training (Optional)

The model can be trained using the training script (if provided). The trained model will be saved as `jingle_detector_model.pkl`.

## Supported Audio Formats

- WAV
- MP3
- OPUS
- OGG
- FLAC
- AAC
- M4A

## Error Handling

- The system provides detailed error messages for unsupported files or processing errors
- Failed files are logged in the output CSV with error details

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Audio File Format Issues**:
   - Ensure the audio files are not corrupted
   - Try converting to a different format (WAV recommended for best compatibility)

3. **Model Loading Errors**:
   - Verify the model file exists at the specified path
   - Ensure the model was trained with a compatible version of scikit-learn