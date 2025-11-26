import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque
import time
import screen_brightness_control as sbc
import pyautogui

# Load trained model
model = load_model("facial_strain_model.h5")

# Webcam
cap = cv2.VideoCapture(0)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Rolling window for stabilizing predictions
PREDICTION_WINDOW = 15
predictions = deque(maxlen=PREDICTION_WINDOW)

# Action throttling
last_state = None
last_action_time = 0
ACTION_INTERVAL = 8  # seconds

def adjust_display(state):
    global last_state, last_action_time

    now = time.time()
    if now - last_action_time < ACTION_INTERVAL:
        return  # cooldown

    if state != last_state:
        if state == "Strained":
            print("😣 Strain detected → dimming brightness + enabling night mode")
            try:
                sbc.set_brightness(40)
                pyautogui.hotkey('winleft', 'ctrl', 'c')  # toggle filter/night mode
            except:
                pass

        else:
            print("🙂 Relaxed detected → restoring brightness + disabling night mode")
            try:
                sbc.set_brightness(80)
                pyautogui.hotkey('winleft', 'ctrl', 'c')
            except:
                pass

        last_state = state
        last_action_time = now


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    label = "No face"

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))

        # Convert grayscale → RGB (model expects 3 channels!)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        pred = model.predict(roi, verbose=0)[0][0]
        predictions.append(pred)

        # stabilised label
        avg_pred = np.mean(predictions)
        label = "Strained" if avg_pred > 0.5 else "Relaxed"

        # Adjust display if needed
        adjust_display(label)

        color = (0, 0, 255) if label == "Strained" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Facial Strain Detection (Stable)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
