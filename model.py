import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class LiveStockModel:
    """
    A class to train a convolutional neural network for livestock health classification.
    
    Attributes:
        file (str): Path to the JSON file containing image paths and labels.
        data (list): List of dictionaries containing image file paths and corresponding labels.
        img_size (tuple): Size to which images will be resized for training.
        num_classes (int): Number of distinct health conditions or diseases.
        label_map (dict): Mapping from label names to numerical indices.
        model (tensorflow.keras.models.Sequential): The CNN model.
        images (list): List of preprocessed images.
        labels (list): List of numerical labels corresponding to the images.

    Methods:
        read_data(): Reads image paths and labels from the JSON file.
        test_training(): Prepares training and validation data from the images.
        create_model(X_train, y_train, X_val, y_val): Defines and trains the CNN model.
        save_model(X_val, y_val): Saves the trained model and evaluates it on validation data.
    """
    
    def __init__(self, file: str) -> None:
        """
        Initializes the LiveStockModel with the given file path.
        
        Args:
            file (str): Path to the JSON file containing image paths and labels.
        """
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

    def read_data(self) -> list[dict]:
        """
        Reads image paths and labels from the JSON file.
        
        Returns:
            list[dict]: List of dictionaries with image paths and labels.
        """
        with open(self.file, 'r') as file:
            data = json.load(file)
        return data

    def test_training(self) -> tuple:
        """
        Prepares training and validation data from the images.
        
        Returns:
            tuple: Contains training and validation data (X_train, X_val, y_train, y_val).
        """
        for entry in self.data:
            img = cv2.imread(entry["filename"])
            if img is None:
                print(f"Warning: Image {entry['filename']} could not be loaded.")
                continue
            img = cv2.resize(img, self.img_size)
            img = img / 255.0
            self.images.append(img)
            self.labels.append(self.label_map[entry["label"]])
        
        X = np.array(self.images)
        y = to_categorical(np.array(self.labels), num_classes=self.num_classes)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        return (X_train, X_val, y_train, y_val)

    def create_model(self, X_train, y_train, X_val, y_val):
        """
        Defines and trains the CNN model using the given training and validation data.
        
        Args:
            X_train (np.ndarray): Training images.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation images.
            y_val (np.ndarray): Validation labels.
        """
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
        """
        Saves the trained model and evaluates it on the validation data.
        
        Args:
            X_val (np.ndarray): Validation images.
            y_val (np.ndarray): Validation labels.
        """
        self.model.save('cow_health_model.keras')

        val_loss, val_accuracy = self.model.evaluate(X_val, y_val)
        print(f'Validation accuracy: {val_accuracy:.2f}')

if __name__ == "__main__":
    model = LiveStockModel('frames.json')
    (X_train, X_val, y_train, y_val) = model.test_training()
    model.create_model(X_train, y_train, X_val, y_val)
    model.save_model(X_val, y_val)
