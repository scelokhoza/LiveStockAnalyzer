import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


with open('frames.json', 'r') as file:
    data = json.load(file)
    

img_size = (128, 128)  
num_classes = len(set(item["label"] for item in data))

label_map = {label: idx for idx, label in enumerate(set(item["label"] for item in data))}

# Prepare the dataset
images = []
labels = []

for entry in data:
    img = cv2.imread(entry["filename"])
    img = cv2.resize(img, img_size)
    img = img / 255.0  
    images.append(img)
    labels.append(label_map[entry["label"]])
    
    
X = np.array(images)
y = to_categorical(np.array(labels), num_classes=num_classes) 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


#end of data processing


#start modelling


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Fit the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          validation_data=(X_val, y_val), 
          epochs=25)

model.save('cow_health_model.h5')

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation accuracy: {val_accuracy:.2f}')