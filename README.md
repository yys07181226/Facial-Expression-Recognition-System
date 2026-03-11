# Facial-Expression-Recognition-System
## Overview
This project implements a Facial Expression Recognition System using YOLO (You Only Look Once) models. The system aims to recognize seven fundamental emotions: Surprise, Fear, Disgust, Happiness, Sadness, Anger, and Neutral.

## Project Structure
* train_YOLO.py: Main script for training the YOLO model.
* emotion_recognition_cam.py: Real-time demonstration script for recognizing facial expressions using a webcam.
* report.pdf: Comprehensive report detailing the project methodology, results, and analysis.
## Installation
### Prerequisites
* Python 3.8 or higher
* Required libraries:
* OpenCV
* NumPy
* PyTorch
* Ultralytics YOLO

## Model Weights
Download the YOLO model weights (e.g., yolov8n.pt, or your custom weights) and place them in the project directory.

## Training the Model
To train the YOLO model, modify the parameters in the train_YOLO.py file according to your dataset and hardware configuration.
### Key Parameters:
* model_path: Path to the pre-trained model.
* data_config: Path to the dataset configuration file.
* output_dir: Directory for saving training results.
* epochs: Number of training epochs.
* batch: Batch size for training.
* imgsz: Input image size for the model.

## Running the Emotion Recognition
run the file demo_emotion_recognition_cam.py

### Key Parameters
* model_path: Path to the trained model weights.
* class_names: List of emotion class names.
* device: Device to run the model ("0" for GPU, "cpu" for CPU).
* cam_id: Camera ID for the input video stream.
* imgsz: Input image size for inference.

## Dataset Characteristics
* Collection: Images are collected from real-world scenarios to ensure diversity in facial expressions.
* Format: The dataset is organized in compliance with the YOLO standard format, enabling easy integration without extensive preprocessing.
* Division:
    * Training Set: Approximately 12,000 samples (77.4%)
    * Testing Set: Approximately 3,500 samples (22.6%)

## Results
Refer to the report.pdf for a detailed analysis of the model's performance, evaluation metrics, and future improvements.







  
