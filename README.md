# Medi-AI

Medi-AI is a skin disease classification web application that helps users analyze images of skin conditions and receive a predicted diagnosis category based on a trained deep learning model.

## Project Overview

This project uses a MobileNetV2-based convolutional neural network to classify common skin conditions from uploaded images. It is designed for educational and prototype use in medical image analysis and can be extended for clinical decision support workflows.

## Supported Classes

The model classifies images into these categories:

- Acne
- Eczema
- Healthy Skin
- Melanoma
- Vitiligo

## Features

- Image upload through a browser interface
- Real-time skin disease prediction
- Confidence score for each prediction
- Disease information cards with symptoms, causes, and recommendations
- Flask-based backend and lightweight web frontend
- MobileNetV2 model trained on a curated skin image dataset

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Flask
- HTML / CSS / JavaScript

## Project Structure

```text
Medi-AI/
├── app.py                 # Flask application and disease information logic
├── predict.py             # Standalone prediction script with model inference
├── train.py               # Model training pipeline
├── requirements.txt       # Python dependencies
├── skin_disease_mobilenetv2.keras  # Trained model file
├── static/                # CSS and JS assets for frontend
├── templates/             # HTML templates
├── README.md              # Project documentation
├── .gitignore             # Ignored local artifacts and large files
└── split_dataset/         # Training/validation/test dataset folders
```

## Model Details

The current model is based on MobileNetV2 with a transfer learning setup. The architecture uses a pre-trained ImageNet backbone and a custom classification head to identify five skin condition classes.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/prabhavi-1700/Medi-Ai.git
cd Medi-Ai
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Flask app:

```bash
python app.py
```

Then open the browser at:

```text
http://127.0.0.1:5000/
```

## Prediction Workflow

- Upload an image of the skin area
- The app validates image quality
- The image is cropped and preprocessed
- The trained model predicts the class
- The result and relevant educational information are shown to the user

## Important Note

This project is intended for research, learning, and prototype demonstration. It should not be used as a substitute for professional medical diagnosis. Any suspicious skin condition should be confirmed by a certified dermatologist.

## Author

Prabhavi Tripathi

## GitHub

- https://github.com/prabhavi-1700
- Repository: https://github.com/prabhavi-1700/Medi-Ai
