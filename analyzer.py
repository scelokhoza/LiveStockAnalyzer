import cv2
import numpy as np
from keras.models import load_model


model = load_model('cow_health_model.h5')


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img = cv2.resize(frame, (128, 128))  
        img = img / 255.0  
        img = np.expand_dims(img, axis=0)  

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = list(label_map.keys())[predicted_class]

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


predictions = analyze_video('path/to/your/video.mp4')
print(predictions)
