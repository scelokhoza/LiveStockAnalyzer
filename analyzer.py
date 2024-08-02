import cv2
import numpy as np
from tensorflow.keras.models import load_model

class AnalyzeLiveStock:
    """
    A class to analyze video frames using a trained livestock health classification model.
    
    Attributes:
        model_file (str): Path to the trained model file.
        video_path (str): Path to the video file to analyze.
        label_map (dict): Mapping from numerical indices to label names.

    Methods:
        analyze_video(): Processes the video frames, makes predictions, and displays results.
    """

    def __init__(self, model_file: str) -> None:
        """
        Initializes the AnalyzeLiveStock with the given model file and video path.
        
        Args:
            model_file (str): Path to the trained model file.
            video_path (str): Path to the video file to analyze.
        """
        self.model = load_model(model_file)
        # self.video_path = video_path
        self.label_map = {
            0: 'healthy',
            1: 'anaplasmosis',
            2: 'bloat',
            3: 'bovine mastitis',
            4: 'cancer eye',
            5: 'johnes disease',
            6: 'ketosis',
            7: 'lumpy skin',
            8: 'mouth disease'            
        }

    def analyze_video(self, video_path):
        """
        Processes the video frames, makes predictions, and displays results.
        
        Returns:
            list: List of predictions for each frame in the video.
        """
        # cap = cv2.VideoCapture(self.video_path)
        # frame_count = 0
        # predictions = []

        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
            
        #     img = cv2.resize(frame, (128, 128))
        #     img = img / 255.0  
        #     img = np.expand_dims(img, axis=0) 

        #     prediction = self.model.predict(img)
        #     predicted_class = np.argmax(prediction, axis=1)[0]
        #     predicted_label = self.label_map.get(predicted_class, 'Unknown')

        #     predictions.append(predicted_label)

        #     cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #     confidence = np.max(prediction) * 100
        #     cv2.putText(frame, f'Confidence: {confidence:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
        #     cv2.imshow('Livestock Monitoring', frame)

        #     # Optional: save to a new video file
        #     # output.write(frame)  # Uncomment if you set up VideoWriter

        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        #     frame_count += 1

        # cap.release()
        # cv2.destroyAllWindows()

        # return predictions
        cap = cv2.VideoCapture(video_path)
        predictions = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img = cv2.resize(frame, (128, 128))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = self.model.predict(img)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = self.label_map.get(predicted_class, 'Unknown')
            predictions.append(predicted_label)

        cap.release()
        return {"message": ", ".join(predictions)}
    
    

if __name__ == '__main__':
    analysis = AnalyzeLiveStock('cow_health_model.keras', 'assets/videos/sickcow.mp4')
    predictions = analysis.analyze_video()
    print(predictions)
