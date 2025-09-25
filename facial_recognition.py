import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict, Optional

class LandmarkDetector():
    def __init__(self, 
                 static_image_mode: bool = False,  
                 max_num_faces: int = 1,  
                 refine_landmarks: bool = True, 
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5): 
        self.mp_face_mesh = mp.solutions.face_mesh      
        self.mp_drawing = mp.solutions.drawing_utils    
        self.mp_drawing_styles = mp.solutions.drawing_styles    
        self.face_mesh = self.mp_face_mesh.FaceMesh(       
            static_image_mode = static_image_mode,
            max_num_faces = max_num_faces,
            refine_landmarks = refine_landmarks,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
            )
        self.define_lms_index()
        

    def define_lms_index(self):
        self.left_eye_indices = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        ]

        self.right_eye_indices = [
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        ]

        self.nose_indices = [6, 456, 360, 455, 460, 462, 242, 240, 235, 131, 236, 6, 1] 
        
        self.mouth_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            78, 95, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88
            ]
        
        self.facial_contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454,
             323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132,
             93, 234, 127, 162, 21, 54, 103, 67, 109]


    def detect_lms(self, image:np.ndarray)->Optional[List]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(image_rgb)
        if result.multi_face_landmarks:
            return result.multi_face_landmarks
        return None
    

    def extract_face_elements(self, landmarks, image_shape: Tuple[int, int])->Dict:
        height, width = image_shape[:2] 
        facial_elements = {
            "left_eye": [],
            "right_eye": [],
            "nose": [],
            "mouth": [],
            "facial_contour":[]
            }

        def _extract_points(indices):
            points = []
            for i in indices:
                if i < len(landmarks.landmark):
                    x = int(landmarks.landmark[i].x * width)
                    y = int(landmarks.landmark[i].y * height)
                    points.append((x, y))
            return points

        facial_elements["left_eye"] = _extract_points(self.left_eye_indices)
        facial_elements["right_eye"] = _extract_points(self.right_eye_indices)
        facial_elements["nose"] = _extract_points(self.nose_indices)
        facial_elements["mouth"] = _extract_points(self.mouth_indices)
        facial_elements["facial_contour"] = self._extract_facial_contour(landmarks, width, height)
        return facial_elements
        
    
    def _extract_facial_contour(self, landmarks, width, height):
        points = []
        offset_a = 22
        offset_b = 17
        for i in self.facial_contour_indices:
            if i < len(landmarks.landmark):
                x = int(landmarks.landmark[i].x * width)
                y = int(landmarks.landmark[i].y * height)
                if i in [67, 109, 10, 338, 297]:
                    y -= offset_a
                if i in [103, 332, 54, 289]:
                    y -= offset_b
                points.append((x, y))
        return points    
    

    def draw_facial_elements(self, image: np.ndarray, facial_elements: Dict)->np.ndarray:
        annotated_img = image.copy()   
        colors =  {
            "left_eye": (0,0,255),    
            "right_eye": (0,0,255),
            "nose": (255,0,0),
            "mouth": (0,255,0),
            "facial_contour": (100,100,100)
            }
        for element, points in facial_elements.items():
            color = colors.get(element, (255,255,255))
            for p in points:  
                cv2.circle(annotated_img, p, 2, color, -1)    
            if element in ["left_eye","right_eye","nose", "mouth", "facial_contour"]:
                pts = np.array(points, np.int32)  
                cv2.polylines(annotated_img, [pts], True, color, 1)
        return annotated_img
    

    def extract_roi(self, facial_elements: Dict, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        x_min, y_min = frame_width, frame_height
        x_max, y_max = 0, 0

        for element, coords in facial_elements.items():
            for (x, y) in coords:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

        margin_x = int(0.2 * (x_max - x_min))
        margin_y = int(0.2 * (y_max - y_min))

        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(frame_width, x_max + margin_x)
        y_max = min(frame_height, y_max + margin_y)

        return x_min, x_max, y_min, y_max


    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Dict, Dict, np.ndarray, int, int]]:
        lms_list = self.detect_lms(frame)  
        facial_elements = {}        
        centers = {}
        roi = frame.copy()
        annotated_frame = frame.copy()
        if not lms_list:
            return None  # Nessun volto rilevato
        lms = lms_list[0] 
        facial_elements = self.extract_face_elements(lms, frame.shape)
        x_min, x_max, y_min, y_max = self.extract_roi(facial_elements, frame.shape[1], frame.shape[0])  
        roi = frame[y_min:y_max, x_min:x_max].copy()
        annotated_frame = self.draw_facial_elements(frame, facial_elements) 

        return annotated_frame, facial_elements, centers, roi, x_min, y_min


