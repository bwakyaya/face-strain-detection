import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURATIONS ---
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32
EPOCHS = 20


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")


if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError(f"Check if 'train' and 'test' folders exist under: {base_dir}")

# --- DATA LOADERS ---
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# --- MODEL DEFINITION ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification: strained vs relaxed
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- TRAINING ---
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS
)

# --- SAVE MODEL ---
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "facial_strain_model.h5"))
model.save(model_path)
print(f"✅ Model saved at {model_path}")
