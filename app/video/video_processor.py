import cv2
import numpy as np
from app.detection.human_detector import HumanDetector
from app.alarm.alarm_handler import AlarmHandler
from app.zone.zone_handler import ZoneHandler


class VideoProcessor:
    def __init__(
        self,
        detector: HumanDetector,
        zone_handler: ZoneHandler,
        alarm_handler: AlarmHandler,
        use_tracking: bool = True,
    ):
        self.detector = detector
        self.zone_handler = zone_handler
        self.alarm_handler = alarm_handler
        self.use_tracking = use_tracking

    def is_point_in_polygon(self, point, polygon):
        return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

    def is_bbox_in_zone(self, bbox, zone):
        """Проверяет, пересекается ли bounding box с запретной зоной"""
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for corner in corners:
            if self.is_point_in_polygon(corner, zone):
                return True
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        return self.is_point_in_polygon(center, zone)

    def draw_detections(self, frame, detections):
        """Рисует обнаружения на кадре"""
        for detection in detections:
            bbox = detection["bbox"]
            track_id = detection.get("track_id", -1)
            confidence = detection["confidence"]

            x1, y1, x2, y2 = bbox
            color = (0, 0, 255) if detection.get("intrusion", False) else (255, 0, 0)
            thickness = 3 if detection.get("intrusion", False) else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{track_id} ({confidence:.2f})"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def process_frame(self, frame):
        """Обрабатывает один кадр"""
        if self.use_tracking:
            detections = self.detector.track_people(frame)
        else:
            detections = self.detector.detect(frame)

        # Проверка пересечения с запретными зонами
        for zone in self.zone_handler.zones:
            for detection in detections:
                if self.is_bbox_in_zone(detection["bbox"], zone):
                    track_id = detection.get("track_id", id(detection))
                    self.alarm_handler.trigger_alarm(track_id)
                    detection["intrusion"] = True

        # Обновление состояния тревоги
        self.alarm_handler.update()

        # Отрисовка
        self.zone_handler.draw_zones(frame)
        self.draw_detections(frame, detections)
        self.alarm_handler.draw_alarm(frame)

        tracking_status = "YOLO Tracking" if self.use_tracking else "No Tracking"
        cv2.putText(
            frame,
            f"People: {len(detections)} | {tracking_status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return frame, detections

    def run(self, video_path: str):
        """Запускает обработку видео"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {video_path}")
            return

        print("Обработка видео... Нажмите 'q' для выхода")
        print("Цветовая кодировка:")
        print("  Красный тонкий - запретные зоны")
        print("  Синий - обнаруженные люди")
        print("  Красный толстый - нарушители в запретной зоне")
        print(f"  Трекинг: {'Включен (YOLO)' if self.use_tracking else 'Отключен'}")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("end file")
                break

            processed_frame, detections = self.process_frame(frame)
            cv2.imshow("Intrusion Detection System", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
