import cv2
import mediapipe as mp
import numpy as np  # Import NumPy for the `where` function

# Initialize MediaPipe solutions for face mesh, pose detection, hand tracking, and segmentation
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_selfie_segmentation = mp.solutions.selfie_segmentation

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face mesh
    face_results = face_mesh.process(rgb_frame)

    # Process pose detection
    pose_results = pose.process(rgb_frame)

    # Process hand tracking
    hand_results = hands.process(rgb_frame)

    # Process segmentation
    segmentation_results = selfie_segmentation.process(rgb_frame)

    # Apply segmentation mask on the frame
    condition = segmentation_results.segmentation_mask > 0.5
    bg_image = np.zeros_like(frame)  # Create a plain black background instead of blur

    # Use np.where instead of cv2.where
    frame = np.where(condition[..., None], frame, bg_image)

    # Draw the face mesh annotations if faces are detected
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # Draw pose landmarks if pose is detected
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=pose_results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

    # Draw hand landmarks if hands are detected
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))

    # Display the output frame
    cv2.imshow('Face Mesh, Pose, Hand Tracking, and Segmentation', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
