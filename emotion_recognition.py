import cv2
from fer import FER
from collections import deque, Counter

detector = FER(mtcnn=True)
last_emotions = deque(maxlen=10)

emotions_map = {
    "angry": "Arrabbiato",
    "disgust": "Digustato",
    "fear": "Impaurito",
    "happy": "Felice",
    "sad": "Triste",
    "surprise": "Sorpreso",
    "neutral": "Neutro"
}

def detect_emotion(frame, annotated_frame, x_offset=0, y_offset=0):
    faces = detector.detect_emotions(frame)

    for face in faces:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)

        last_emotions.append(dominant_emotion)
        most_common_emo = Counter(last_emotions).most_common(1)[0][0]

        emo_it = emotions_map.get(most_common_emo, most_common_emo)

        x += x_offset
        y += y_offset

        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated_frame, emo_it,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
        print(f"Emozione rilevata: {emo_it}")
    return annotated_frame


