import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout




class LiveStockModel:
    def __init__(self, file: str) -> None:
        self.file = file
        self.data = self.read_data()
        self.img_size = (128, 128)  
        self.num_classes = len(set(item["label"] for item in self.data))
        self.label_map = {label: idx for idx, label in enumerate(set(item["label"] for item in self.data))}
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        self.images = []
        self.labels = []

        
        
    def read_data(self) -> list[dict, dict]:
        with open(self.file, 'r') as file:
            data = json.load(file)
        return data
    
    
    def test_training(self) -> tuple:
    
        for entry in self.data:
            img = cv2.imread(entry["filename"])
            img = cv2.resize(img, self.img_size)
            img = img / 255.0  
            self.images.append(img)
            self.labels.append(self.label_map[entry["label"]])
        
        X = np.array(self.images)
        y = to_categorical(np.array(self.labels), num_classes=self.num_classes) 

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return (X_train, X_val, y_train, y_val)

    def create_model(self, X_train, y_train, X_val, y_val):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        self.model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                validation_data=(X_val, y_val), 
                epochs=25)
    
    def save_model(self, X_val, y_val):
        self.model.save('cow_health_model.h5')

        val_loss, val_accuracy = self.model.evaluate(X_val, y_val)
        print(f'Validation accuracy: {val_accuracy:.2f}')
        
        

if __name__ == "__main__":
    model = LiveStockModel('frames.json')
    (X_train, X_val, y_train, y_val) = model.test_training()
    model.create_model(X_train, y_train, X_val, y_val)
    model.save_model(X_val, y_val)