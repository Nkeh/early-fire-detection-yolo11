import os
from ultralytics import YOLO

def train_model(data_yaml_path, model_name='yolov11n.pt', epochs=100, imgsz=640, batch_size=16, project='runs/train', name='fire_detection'):
    """
    Train a YOLOv11 model on the provided dataset.

    Args:
        data_yaml_path (str): Path to the data.yaml file containing dataset configuration.
        model_name (str): Pre-trained YOLOv11 model to use (e.g., 'yolov11n.pt', 'yolov11s.pt').
        epochs (int): Number of training epochs.
        imgsz (int): Image size for training.
        batch_size (int): Batch size for training.
        project (str): Directory to save training results.
        name (str): Name of the training run.

    Returns:
        None
    """
    # Check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    # Load the YOLO model
    model = YOLO(model_name)

    # Train the model
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project,
        name=name,
        save=True,  # Save the best model
        save_period=10,  # Save model every 10 epochs
        cache=True,  # Cache images for faster training
        device='auto'  # Use GPU if available, else CPU
    )

    print(f"Training completed. Results saved in {os.path.join(project, name)}")

if __name__ == "__main__":
    # Example usage
    data_path = "Fire-Smoke-Detection-Yolov11-1/data.yaml"
    train_model(data_path)