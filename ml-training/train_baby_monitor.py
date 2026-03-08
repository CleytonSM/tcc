from ultralytics import YOLO

def main():
    # Load model
    model = YOLO("yolo26n.pt")

    model.train(
        data="ml-training/baby_monitor.yaml",
        epochs=100,
        imgsz=640,
        patience=50,
        batch=16
    )
    metrics = model.val()

if __name__ == "__main__":
    main()