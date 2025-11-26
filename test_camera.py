import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully!")
    print("Press 'q' on the video window to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)

        cv2.imshow("Webcam - Press 'q' to exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
