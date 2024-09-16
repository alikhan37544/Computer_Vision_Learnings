import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8n.pt' for a smaller model

# Initialize variables for motion detection
first_frame = None
trackers = []  # List to hold trajectory points

# Initialize video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize the first frame for comparison
    if first_frame is None:
        first_frame = gray
        continue

    # Calculate the absolute difference between the current frame and the first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Contours for the thresholded image to identify regions of change
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for areas of change
    mask = np.zeros_like(frame)

    # Draw the detected areas (regions of interest) on the mask
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue  # Ignore small changes
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

    # Apply the mask to the frame for YOLOv8 detection only in the region of motion
    masked_frame = cv2.bitwise_and(frame, mask)

    # Detect objects using YOLOv8 (only in the masked region)
    results = model(masked_frame, conf=0.5)  # Run YOLOv8 on the masked frame

    # Process the results (bounding boxes and labels)
    for result in results:
        for bbox in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw bounding box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Track the object trajectory
            trackers.append(center)
            if len(trackers) > 50:
                trackers.pop(0)  # Limit trajectory length

    # Draw trajectory on the frame
    for i in range(1, len(trackers)):
        if trackers[i - 1] is None or trackers[i] is None:
            continue
        cv2.line(frame, trackers[i - 1], trackers[i], (0, 0, 255), 2)

    # Show threshold image and the output frame with motion tracking
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Motion Detection and Tracking", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
