from ultralytics import YOLO
import cv2


model = YOLO("best.pt")
video = cv2.VideoCapture("baby.mp4")

# Create a window that can be resized
cv2.namedWindow("Detection Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection Window", 800, 600)  # Moderate size

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Run inference with verbose=False to keep the console clean
    results = model(frame, verbose=False)

    # The easy way to visualize detections is to use the plot() method
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("Detection Window", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()