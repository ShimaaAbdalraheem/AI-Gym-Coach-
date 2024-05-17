import os
import cv2
import mediapipe as mp
import numpy as np

class DataPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def preprocess_data(self, resize_shape=(224, 224)):
        X = []
        y = []
        exercise_labels = {}  # Mapping of exercise names to numerical labels
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Get list of subdirectories (each representing an exercise)
        exercises = sorted(os.listdir(self.data_dir))
        for i, exercise in enumerate(exercises):
            exercise_labels[exercise] = i
            exercise_dir = os.path.join(self.data_dir, exercise)
            for filename in os.listdir(exercise_dir):
                img_path = os.path.join(exercise_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, resize_shape)  # Resize image
                    # Process image using MediaPipe Pose Detection
                    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = pose.process(image_rgb)
                    if result.pose_landmarks:
                    # Extract pose landmarks
                        landmarks = [[lm.x, lm.y] for lm in result.pose_landmarks.landmark]
                        X.append(landmarks)
                        y.append(i)

        X = np.array(X)
        y = np.array(y)
        return X, y, exercise_labels

# usage
data_dir = "exercise_data"
preprocessor = DataPreprocessor(data_dir)
X, y, exercise_labels = preprocessor.preprocess_data()
print("Number of images:", len(X))
print("Labels:", exercise_labels)
