import cv2
import numpy as np
import joblib
import pyttsx3
import os

# -------- CONFIG --------

MODEL_PATH = "models/gesture_model.pkl"
IMG_SIZE = (64, 64)

ROI_TOP = 50
ROI_BOTTOM = 350
ROI_LEFT = 50
ROI_RIGHT = 350

# Initialize TTS engine ONCE globally
tts_engine = pyttsx3.init()


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model '{MODEL_PATH}' not found. Train the model first.")
        return None
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
    return model


def speak(text: str):
    """
    Speak the given text using pyttsx3.
    Does nothing if text is empty.
    """
    text = text.strip()
    if not text:
        print("Nothing to speak.")
        return

    try:
        # stop anything in the queue, then say the new text
        tts_engine.stop()
        tts_engine.say(text)
        tts_engine.runAndWait()
    except KeyboardInterrupt:
        # If user interrupts while speaking, just return gracefully
        print("Speech interrupted by user.")
    except Exception as e:
        print(f"TTS error: {e}")


def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    return resized.flatten()


def main():
    model = load_model()
    if model is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_pred = ""
    stable_frames = 0
    STABLE_REQUIRED = 5

    sentence = ""  # Holds final sentence text

    print("Webcam opened. Place your hand inside the green box.")
    print("Controls: S = Speak, C = Clear, Q = Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)

        # Draw ROI
        cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)
        roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]

        # Gesture prediction
        if roi.size != 0:
            features = preprocess_roi(roi)
            pred = model.predict([features])[0]

            if pred == last_pred:
                stable_frames += 1
            else:
                stable_frames = 0
                last_pred = pred

            # Only add when prediction is stable
            if stable_frames >= STABLE_REQUIRED:
                # Avoid repeating the same word continuously
                if not sentence.endswith(pred + " "):
                    sentence += pred + " "

        # Display sentence and controls
        cv2.putText(frame, f"Sentence: {sentence}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, "S=Speak  C=Clear  Q=Quit", (20, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Sign Recognition + Speech", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            speak(sentence)

        if key == ord('c'):
            sentence = ""

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
