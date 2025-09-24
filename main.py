import cv2
from facial_recognition import LandmarkDetector
from emotion_recognition import detect_emotion
import sys
from PIL import Image, ImageOps
import numpy as np


def resize_with_pil(frame, max_width=640, max_height=480):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    resized = ImageOps.contain(img, (max_width, max_height))
    return cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)


def sheldonscope(source_type="webcam", source_path=None):
    detector = LandmarkDetector()

    if source_type == "webcam":
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

    elif source_type == "video":
        if source_path is None:
            print("Error: video path required")
            return

        capture = cv2.VideoCapture(source_path)
        if not capture.isOpened():
            print("Error: unable to open the video")
            return

        print("Press 'q' to exit")
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame = resize_with_pil(frame, 640, 480)

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

    elif source_type == "image":
        if source_path is None:
            print("Error: image path required")
            return

        frame = cv2.imread(source_path)
        if frame is None:
            print("Error: image not found or unreadable")
            return

        frame = resize_with_pil(frame, 640, 480)

        result = detector.process_frame(frame)
        if result is None:
            cv2.imshow("Sheldonscope", frame)
        else:
            annotated_frame, facial_elements, centers, roi, x_min, y_min = result
            annotated_frame = detect_emotion(roi, annotated_frame, x_min, y_min)
            cv2.imshow("Sheldonscope", annotated_frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Error: invalid source type")


if __name__ == "__main__":
    # Esempi d’uso:
    # python main.py                 -> webcam
    # python main.py video file.mp4  -> video
    # python main.py image foto.jpg  -> immagine
    args = sys.argv[1:]
    if len(args) == 0:
        sheldonscope("webcam")
    elif args[0] == "video":
        sheldonscope("video", args[1])
    elif args[0] == "image":
        sheldonscope("image", args[1])
    else:
        print("Argomento non valido. Usa: webcam | video | image")
