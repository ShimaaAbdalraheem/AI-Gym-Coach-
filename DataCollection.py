import cv2
import os

class DataCollector:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.exercises = {'stationary lunges': ['MyDataSetFitness2/stationary lunges/Dumbbell_Stationary_Lunge2.mp4','MyDataSetFitness2/stationary lunges/STATIONARY LUNGES3.mp4','MyDataSetFitness2/stationary lunges/Stationary_Lunge_with_Dumbbells1.mp4'],'Wrong stationary lunges':['MyDataSetFitness2/Wrong stationary lunges/Dumbbell_Stationary_Lunges_false3.mp4','MyDataSetFitness2/Wrong stationary lunges/stationary_lunges_false.mp4']}  # Dictionary of exercises and corresponding video file paths

    def collect_data(self, num_samples_per_exercise=1000):
        for exercise, videos in self.exercises.items():
            os.makedirs(os.path.join(self.output_dir, exercise), exist_ok=True)
            print(f"Collecting data for {exercise}...")
            
            total_frames = 0  # Total frames collected across all videos for the current exercise
            
            for video_path in videos:
                cap = cv2.VideoCapture(video_path)  # Capture video from video_path
                
                for i in range(total_frames, total_frames + num_samples_per_exercise):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Save frame as image
                    img_path = os.path.join(self.output_dir, exercise, f"{exercise}_{i}.jpg")
                    cv2.imwrite(img_path, frame)
                    cv2.imshow('Frame', frame)
                   
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                total_frames += num_samples_per_exercise  # Update total_frames for the next video
                cap.release()
        cv2.destroyAllWindows()
        print("Data collection complete.")

# Example usage
output_directory = "exercise_data"
collector = DataCollector(output_directory)
collector.collect_data(num_samples_per_exercise=1000)
