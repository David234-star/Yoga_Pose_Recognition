import cv2
import mediapipe as mp
import pandas as pd

# Load MediaPipe pose tracking model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load video capture
cap = cv2.VideoCapture(1)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image to detect poses
        results = pose.process(image)

        # Convert the image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Extract pose landmark coordinates and store in DataFrame

            landmarks = []
            for point in mp_pose.PoseLandmark:
                normalized_landmark = results.pose_landmarks.landmark[point]
                landmarks.append(normalized_landmark.x)
                landmarks.append(normalized_landmark.y)
                landmarks.append(normalized_landmark.z)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.DrawingSpec(
                    color=(251, 245, 202), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing_styles.DrawingSpec(
                    color=(251, 245, 202), thickness=3),
                # Uses the same random color for both landmarks and connections
            )

            print(len(landmarks))

            # Save pose tracking data to a CSV file
            data = str(landmarks)
            data = data[1:-1]  # removing square brackets of list
            f = open('Nothing.csv', 'a')
            f.write(str(data)+',Nothing doing\n')
            f.close()

        cv2.imshow('Yoga Tracking', image)

        # Press 'q' to exit
        if cv2.waitKey(10) in [27, ord('q')]:
            break

# Release resources
cap.release()
