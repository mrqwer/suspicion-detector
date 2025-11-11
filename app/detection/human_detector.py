from ultralytics import YOLO
import cv2


def track(video_path):
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    ret = True

    while ret:
        ret, frame = cap.read()

        results = model.track(frame, persist=True)

        frame_ = results[0].plot()
        cv2.imshow("frame", frame_)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break


# class HumanDetector:
#     def __init__(self, model_path: str):
#         self.model = YOLO(model_path)

#     def detect_people(self, frame):
#         results = self.model(frame)
#         detections = []
#         for r in results[0].boxes:
#             if int(r.cls[0]) == 0:
#                 x1, y1, x2, y2 = map(int, r.xyxy[0])
#                 detections.append(((x1, y1, x2, y2)))
#         return detections w
