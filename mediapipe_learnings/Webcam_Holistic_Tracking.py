import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start webcam capture
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image to get holistic results
        results = holistic.process(rgb_frame)

        # Draw face landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.face_landmarks,
                connections=mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.pose_landmarks,
                connections=mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        # Draw hand landmarks (left and right)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.left_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.right_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

        # Display the output frame
        cv2.imshow('Holistic Tracking', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
