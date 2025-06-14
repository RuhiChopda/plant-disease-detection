import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
MODEL_PATH = 'model/plant_disease_model.h5'
LABELS_PATH = 'model/labels.npy'

# Constants
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

# Create model directory if not exists
os.makedirs('model', exist_ok=True)

# Get class names
class_names = sorted(os.listdir(TRAIN_DIR))
print(f"Classes: {class_names}")

# Save class names for later prediction
np.save(LABELS_PATH, class_names)

# Image Data Generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Build Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# Compile
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)

# Save
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
