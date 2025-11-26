import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


DATASET_DIR = "Dataset"
MODEL_PATH = "models/gesture_model.pkl"

IMG_SIZE = (64, 64)


def load_dataset():
    """
    Loads images from Dataset folder.
    Assumes structure:
      Dataset/
        HELLO/
        YES/
        ...
    Returns: X (features), y (labels), label_names (list of class names)
    """
    X = []
    y = []
    label_names = []

    if not os.path.exists(DATASET_DIR):
        print(f"Dataset folder '{DATASET_DIR}' not found!")
        return None, None, None

    for label in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder_path):
            continue

        label_names.append(label)
        print(f"Loading images for label: {label}")

        # For each image in the folder
        for filename in os.listdir(folder_path):
            if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
                continue

            img_path = os.path.join(folder_path, filename)

            # Read image
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gray_resized = cv2.resize(gray, IMG_SIZE)

            feature_vector = gray_resized.flatten()

            X.append(feature_vector)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Total samples loaded: {len(y)}")
    return X, y, label_names


def train_model(X, y):
    """
    Train a RandomForestClassifier on the dataset.
    Splits data into train and test, prints accuracy and report.
    Saves the trained model to disk.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training model...")
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    clf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {acc * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def main():
    X, y, label_names = load_dataset()

    if X is None or len(X) == 0:
        print("No data found. Please capture images first.")
        return

    print("Labels found:", np.unique(y))
    train_model(X, y)

if __name__ == "__main__":
    main()