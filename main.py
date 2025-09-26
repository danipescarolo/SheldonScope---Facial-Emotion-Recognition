import cv2
import sys
import os
from PIL import Image, ImageOps
import numpy as np
from facial_recognition import LandmarkDetector
from emotion_recognition import detect_emotion

# Creazione della cartella output solo se siamo in Docker
if os.path.exists("/.dockerenv"):
    os.makedirs("output", exist_ok=True)


def resize_with_pil(frame, max_width=640, max_height=480):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    resized = ImageOps.contain(img, (max_width, max_height))
    return cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)


def can_show_gui():
    return os.environ.get("DISPLAY") is not None or sys.platform.startswith("win")


def safe_imshow(window_name, frame, output_path="output/output.jpg"):
    if can_show_gui():
        cv2.imshow(window_name, frame)
    else:
        # Salva il frame solo se la cartella esiste
        if os.path.exists(os.path.dirname(output_path)):
            cv2.imwrite(output_path, frame)
            print(f"[INFO] Output salvato in {output_path}")
        else:
            print(f"[INFO] La cartella {os.path.dirname(output_path)} non esiste, frame non salvato.")


def sheldonscope(source_type="webcam", source_path=None):
    detector = LandmarkDetector()

    if source_type == "webcam":
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("Error: unable to open the webcam")
            return

        print("Press 'q' to exit (solo locale)")
        frame_idx = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Error: unable to read the frame")
                break

            result = detector.process_frame(frame)

            if result is None:
                safe_imshow("Sheldonscope", frame, f"output/frame_{frame_idx}.jpg")
            else:
                annotated_frame, facial_elements, centers, roi, x_min, y_min = result
                annotated_frame = detect_emotion(roi, annotated_frame, x_min, y_min)
                safe_imshow("Sheldonscope", annotated_frame, f"output/frame_{frame_idx}.jpg")

            frame_idx += 1

            if can_show_gui() and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        capture.release()
        if can_show_gui():
            cv2.destroyAllWindows()

    elif source_type == "video":
        if source_path is None:
            print("Error: video path required")
            return

        capture = cv2.VideoCapture(source_path)
        if not capture.isOpened():
            print("Error: unable to open the video")
            return

        print("Press 'q' to exit (solo locale)")
        frame_idx = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame = resize_with_pil(frame, 640, 480)

            result = detector.process_frame(frame)
            if result is None:
                safe_imshow("Sheldonscope", frame, f"output/video_frame_{frame_idx}.jpg")
            else:
                annotated_frame, facial_elements, centers, roi, x_min, y_min = result
                annotated_frame = detect_emotion(roi, annotated_frame, x_min, y_min)
                safe_imshow("Sheldonscope", annotated_frame, f"output/video_frame_{frame_idx}.jpg")

            frame_idx += 1

            if can_show_gui() and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        capture.release()
        if can_show_gui():
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
            safe_imshow("Sheldonscope", frame, "output/output.jpg")
        else:
            annotated_frame, facial_elements, centers, roi, x_min, y_min = result
            annotated_frame = detect_emotion(roi, annotated_frame, x_min, y_min)
            safe_imshow("Sheldonscope", annotated_frame, "output/output.jpg")

        if can_show_gui():
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
    elif args[0] == "video" and len(args) > 1:
        sheldonscope("video", args[1])
    elif args[0] == "image" and len(args) > 1:
        sheldonscope("image", args[1])
    else:
        print("Argomento non valido. Usa: webcam | video <file> | image <file>")
