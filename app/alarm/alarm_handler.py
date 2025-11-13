import time
import cv2


class AlarmHandler:
    def __init__(self, alarm_duration: int = 3):
        self.alarm_active = False
        self.alarm_start_time: None | float = None
        self.alarm_duration = alarm_duration
        self.tracked_intrusions = {}  # {track_id: last_intrusion_time}

    def trigger_alarm(self, track_id: int = None):
        """Активируем сигнал тревоги"""
        current_time = time.time()

        if track_id is not None:
            self.tracked_intrusions[track_id] = current_time

        if not self.alarm_active:
            self.alarm_active = True
            self.alarm_start_time = current_time

    def update(self):
        """Обновляем состояние тревоги"""
        current_time = time.time()

        # Проверяем индивидуальные тревоги
        active_intrusions = False
        for track_id in list(self.tracked_intrusions.keys()):
            if current_time - self.tracked_intrusions[track_id] <= self.alarm_duration:
                active_intrusions = True
            else:
                del self.tracked_intrusions[track_id]

        # Обновляем общее состояние тревоги
        if active_intrusions:
            self.alarm_active = True
            self.alarm_start_time = current_time
        elif (
            self.alarm_active
            and current_time - self.alarm_start_time > self.alarm_duration
        ):
            self.alarm_active = False

    def draw_alarm(self, frame):
        """Отображаем сигнал тревоги на кадре"""
        if self.alarm_active:
            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                "ALARM!",
                (w // 2 - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

            # Для обращение внимания добавляем красную рамку вокруг всего видео
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
