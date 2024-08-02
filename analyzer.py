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
        cap = cv2.VideoCapture(video_path)
        output_path = 'static/output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

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

            # Draw bounding box around the detected area (dummy box for example)
            cv2.rectangle(frame, (50, 50), (300, 300), (0, 255, 0), 2)

            # Display the predicted label
            cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the confidence score
            confidence = np.max(prediction) * 100
            cv2.putText(frame, f'Confidence: {confidence:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Write the frame into the output file
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return {'output_path': output_path, 'predictions': predictions}
    
    

if __name__ == '__main__':
    analysis = AnalyzeLiveStock('cow_health_model.keras', 'assets/videos/sickcow.mp4')
    predictions = analysis.analyze_video()
    print(predictions)
