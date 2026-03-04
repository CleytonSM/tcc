from ultralytics import YOLO
import torch
import torch_directml

def main():
    # Load model
    model = YOLO("yolov8n.pt")

    model.train(data="ml-training/baby_monitor.yaml", epochs=30)
    metrics = model.val()

if __name__ == "__main__":
    main()