from ultralytics import YOLO
import os

def train_model():
    # Correct Windows path to your data.yaml
    data_yaml_path = r'C:\Users\aanub\Documents\Project2\data\data.yaml'
    
    # Verify the file exists
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    # Load model (choose yolov8n/s/m/l/x)
    model = YOLO('yolov8n.pt')  # or 'yolov8s.pt', etc.

    # Train the model
    results = model.train(
        data=data_yaml_path,  # Using the full Windows path
        epochs=50,
        batch=16,
        imgsz=640,
        patience=5,
        device='0',  # '0' for GPU, 'cpu' for CPU
        name='traffic_violation_v1'
    )

    # Print where the trained model is saved
    print(f"\nTraining complete! Model saved to: {os.path.abspath('runs/detect/traffic_violation_v1/weights/best.pt')}")

if __name__ == '__main__':
    train_model()