import cv2
import os


DATASET_DIR = "Dataset"

GESTURES = {
    "1": "HELLO",
    "2": "YES",
    "3": "NO",
    "4": "THANKYOU",
    "5": "STOP"
}

IMAGE_SIZE = (224, 224)

ROI_TOP = 50      
ROI_BOTTOM = 350  
ROI_LEFT = 50     
ROI_RIGHT = 350   


def create_directories():
    """
    Create Dataset main folder and subfolders for each gesture, if they don't exist.
    """
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    for label in GESTURES.values():
        folder_path = os.path.join(DATASET_DIR, label)
        os.makedirs(folder_path, exist_ok=True)


def get_image_count():
    """
    Count how many images already exist for each label.
    This helps to continue from the last number if you restart the script.
    """
    counts = {}
    for label in GESTURES.values():
        folder_path = os.path.join(DATASET_DIR, label)
        if os.path.exists(folder_path):
            counts[label] = len([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(".jpg") or f.lower().endswith(".png")
            ])
        else:
            counts[label] = 0
    return counts


def main():
    create_directories()

    image_counts = get_image_count()
    print("Current image counts:", image_counts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully!")
    print("Instructions:")
    print(" - Put your hand inside the green rectangle.")
    print(" - Press keys 1-5 to capture an image for a gesture:")
    for key, label in GESTURES.items():
        print(f"   {key} -> {label}")
    print(" - Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)
        cv2.putText(frame, "Place hand inside box", (ROI_LEFT, ROI_TOP - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y0 = 30
        dy = 25
        cv2.putText(frame, "Press 1-5 to capture:", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        i = 1
        for key, label in GESTURES.items():
            text = f"{key}: {label} ({image_counts[label]})"
            cv2.putText(frame, text, (10, y0 + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            i += 1

        cv2.putText(frame, "Press 'q' to quit", (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Capture Hand Images", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if chr(key) in GESTURES:
            label = GESTURES[chr(key)]
            roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]

            roi_resized = cv2.resize(roi, IMAGE_SIZE)

            count = image_counts[label] + 1
            filename = f"{label}_{count:04d}.jpg"
            save_path = os.path.join(DATASET_DIR, label, filename)

            cv2.imwrite(save_path, roi_resized)
            image_counts[label] = count

            print(f"Saved {save_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
