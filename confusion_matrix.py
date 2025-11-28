from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = load_model('models/facial_strain_model.h5')

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/test',  # Path to test data
    target_size=(48, 48),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Important for correct label alignment
)

# Predict
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Relaxed", "Strained"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Strain Detection")
plt.show()

# Optional: Print raw numbers
print("Confusion Matrix:")
print(cm)
