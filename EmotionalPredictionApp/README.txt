
# Audio Emotion Recognition

This project is an audio emotion recognition system built using deep learning, leveraging a combination of Conv1D and LSTM layers to classify emotions from speech audio samples. The application includes a Streamlit-based interactive interface to upload audio files and predict emotions.

## Table of Contents
- Overview
- Dataset
- Project Structure
- Installation
- Virtual Environment Setup
- Usage
- Contributing
- License

---

## Overview
This project uses the **TESS Toronto emotional speech set** as the dataset to classify emotions in audio samples. It trains a deep learning model using features extracted from audio files, specifically MFCC (Mel Frequency Cepstral Coefficients). The final application is deployed with Streamlit, enabling easy user interaction.

## Dataset
The **TESS (Toronto Emotional Speech Set)** dataset contains 2800 audio samples across multiple emotions such as fear, happiness, sadness, anger, and surprise. Each file is labeled according to its emotion, which is extracted and used to train the model.

Download the dataset from the [official TESS repository](https://tspace.library.utoronto.ca/handle/1807/24487) and place it in the project directory.

## Project Structure
```
Audio Emotion Recognition/
├── Emotional_speech_recognition_75721200059.py  # Main training and evaluation script
├── emotion_prediction_app1.py                   # Streamlit application script
├── optimized_model.h5                           # Trained model
├── encoder.joblib                               # Trained encoder for label encoding
├── requirements.txt                             # Required Python libraries
├── README.md                                    # Project documentation
└── ENV/                                         # Virtual environment (optional)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Virtual Environment Setup
It is recommended to create a virtual environment to avoid dependency conflicts.

1. Create a virtual environment:
   ```bash
   python -m venv ENV
   ```

2. Activate the virtual environment:

   - **Windows**
     ```bash
     .\ENV\Scriptsctivate
     ```
   - **MacOS/Linux**
     ```bash
     source ENV/bin/activate
     ```

3. Install dependencies within the virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train and Evaluate the Model
Run the main training script to train and evaluate the model:
```bash
python Emotional_speech_recognition_75721200059.py
```

### Run the Streamlit Application
To start the Streamlit app, run the following command:
```bash
streamlit run emotion_prediction_app1.py
```

After launching, upload an audio file to predict its emotion. The app supports MP3 and WAV file formats.

## Contributing
Feel free to submit pull requests for improvements or new features.

## License
This project is licensed under the MIT License.
