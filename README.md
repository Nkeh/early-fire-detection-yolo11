# Fire and Smoke Detection System

A computer vision project implementing fire and smoke detection using YOLOv11 models through Roboflow's inference API.

## What I Built

This project demonstrates practical application of deep learning for real-time object detection in computer vision. I created a complete pipeline that can:

- **Process Images**: Detect fire and smoke in JPG/PNG images
- **Process Videos**: Analyze video files and generate annotated output videos
- **Batch Processing**: Automatically handle multiple files from input directory
- **Model Training**: Train custom YOLOv11 models on fire/smoke datasets
- **Visualization**: Generate bounding boxes and detection counts on processed media

## Project Structure

```
├── main.py              # Batch processing script for images/videos
├── train.py             # YOLOv11 model training
├── predict.py           # Single image inference
├── fire_detection.ipynb # Interactive notebook experiments
├── predict/input/       # Input images/videos
├── predict/output/      # Processed results
└── .env                 # API configuration
```

## Technologies Used

- **YOLOv11**: State-of-the-art object detection model
- **Roboflow**: Dataset management and inference API
- **OpenCV**: Computer vision operations
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO implementation

## Setup & Usage

1. **Environment**: Create virtual environment and install dependencies
2. **Configuration**: Set up `.env` with Roboflow API credentials
3. **Input**: Place images/videos in `predict/input/`
4. **Run**: Execute `python main.py` for batch processing
5. **Output**: Annotated results saved to `predict/output/`

## Key Learning Outcomes

- Implemented end-to-end computer vision pipeline
- Integrated cloud-based ML inference APIs
- Handled both image and video processing workflows
- Applied deep learning models for real-world detection tasks
- Managed batch processing and file I/O operations

## Results

The system successfully processes input media and generates visual detections with confidence scores, demonstrating practical application of modern computer vision techniques.