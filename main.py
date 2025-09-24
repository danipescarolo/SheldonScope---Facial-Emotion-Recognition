import cv2
from facial_recognition import LandmarkDetector
from emotion_recognition import detect_emotion


def sheldonscope():
    detector = LandmarkDetector()
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Error: unable to open the webcam")
        return

    print("Press 'q' to exit")
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: unable to read the frame")
            break

        result = detector.process_frame(frame)

        if result is None:
            cv2.imshow("Sheldonscope", frame)
        else:
            annotated_frame, facial_elements, centers, roi, x_min, y_min = result
            annotated_frame = detect_emotion(roi, annotated_frame, x_min, y_min)
            cv2.imshow("Sheldonscope", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sheldonscope()
