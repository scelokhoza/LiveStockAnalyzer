import cv2
import numpy as np
from tensorflow.keras.models import load_model

class AnalyzeLiveStock:
    def __init__(self, model_file: str, video_path) -> None:
        self.model = load_model(model_file)
        self.video_path = video_path
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

    def analyze_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
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

            cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Livestock Monitoring', frame)

            # Optional: save to a new video file
            # output.write(frame)  # Uncomment if you set up VideoWriter

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        return predictions
    

if __name__ == '__main__':
    analysis = AnalyzeLiveStock('cow_health_model.h5', 'sickcow.mp4')
    predictions = analysis.analyze_video()
    print(predictions)
