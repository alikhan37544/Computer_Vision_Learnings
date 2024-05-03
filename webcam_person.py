from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)  # Open webcam
cap.set(3, 640)  # Set width to 640 pixels
cap.set(4, 480)  # Set height to 480 pixels

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Load YOLOv8n model

# Get person class index
person_index = model.names.get("person")  # Get index of "person" class

while True:
    success, img = cap.read()  # Read frame from webcam

    if not success:
        break  # Exit if frame reading fails

    results = model(img, stream=True)  # Run inference on the frame

    # Process detected objects
    for r in results:
        boxes = r.boxes  # Get bounding boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers

            # Check if class is "person"
            cls = int(box.cls[0])
            if cls == person_index:  # Only process "person" class

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    # Display image with bounding boxes
    cv2.imshow('Webcam', img)

    # Quit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
