# Traffic Violation Detection using YOLO

This project detects vehicles violating traffic rules using YOLO (You Only Look Once) object detection. It can identify vehicles from images/videos, highlight violations, and extract license plates for further action. The project includes both training and inference pipelines, along with a simple application interface for demonstrations.

## 🚦 Features
- Detect vehicles in real-time or from files
- Identify traffic violations (e.g., signal jumping, lane crossing)
- Extract vehicle number plates
- Supports multiple YOLO models (`yolov8n`, `yolov8s`, `yolo11n`)
- Easy-to-use app interface

## 📂 Project Structure
- `train.py` → Train YOLO model on custom dataset
- `script.py` → Run detection & save results
- `app.py` → Interactive app (Streamlit)
- `data/` → Dataset folder
- `runs/detect/` → Output results

## 🛠 Installation
```bash
git clone https://github.com/Anubhav-Anand-26/Minor-Project-2.git
cd Minor-Project-2
pip install -r requirements.txt
