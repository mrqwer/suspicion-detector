from ultralytics import YOLO
import cv2
import numpy as np


class HumanDetector:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        self.model = YOLO(model_path)
        self.device = device
        self.class_names = ["person"]

    def detect(self, frame):
        """Обнаружение людей в кадре"""
        # предиктим переданный кадр
        results = self.model.predict(
            source=frame,
            device=self.device,
            verbose=False,  # Отключаем вывод в консоль
            classes=[0],  # Берем только людей (class_id=0)
            conf=0.5,  # Порог уверенности
            imgsz=640,  # Размер изображения
        )
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id == 0 and confidence > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": confidence,
                                "class_name": "person",
                            }
                        )
        return detections

    def track_people(self, frame):
        """Трекинг людей с использованием встроенного трекера YOLO"""
        results = self.model.track(
            frame,
            device=self.device,
            verbose=False,
            classes=[0],
            conf=0.5,
            persist=True,  # Продолжать трекинг между кадрами
            imgsz=640,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id == 0 and confidence > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detection = {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_name": "person",
                        }

                        # Добавляем track_id если есть
                        if hasattr(box, "id") and box.id is not None:
                            detection["track_id"] = int(box.id)

                        detections.append(detection)

        return detections
