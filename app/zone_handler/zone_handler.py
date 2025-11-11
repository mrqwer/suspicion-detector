import cv2
import json
import os
import numpy as np


class ZoneHandler:
    def __init__(self, zones_path: str = "restricted_zones.json"):
        self.zones_path = zones_path
        self.zones = []  # список уже сохранённых зон
        self.current_zone = []  # зона, которую рисуем сейчас
        self.window_name = "Определить запрещенную зону"

        # Если файл с зонами уже есть — загружаем
        if os.path.exists(zones_path):
            with open(zones_path, "r", encoding="utf-8") as f:
                self.zones = json.load(f)

    def draw_zones(self, frame):
        """Рисуем все сохранённые и текущие зоны на кадре."""
        # Красные — уже сохранённые зоны
        for zone in self.zones:
            pts = [tuple(pt) for pt in zone]
            cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 0, 255), 2)

        # Зелёная — зона, которую пользователь сейчас рисует
        if len(self.current_zone) > 1:
            cv2.polylines(
                frame, [np.array(self.current_zone, np.int32)], False, (0, 255, 0), 2
            )

    def select_zone(self, video_path: str):
        """Берём первый кадр из видео и даём пользователю мышкой отметить зону."""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print("Не удалось считать видео")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Центрируем окно (чтобы не открывалось где попало)
        try:
            import tkinter as tk

            root = tk.Tk()
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
        except Exception:
            sw, sh = 1920, 1080  # запасной вариант

        fh, fw = frame.shape[:2]
        cv2.moveWindow(self.window_name, (sw - fw) // 2, (sh - fh) // 2)

        # Подключение обработчика кликов
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("Надо кликать ЛКМ, чтобы добавить точки зоны.")
        print("'s' чтобы сохранить, 'q' чтобы выйти без сохранения.")

        while True:
            temp = frame.copy()
            self.draw_zones(temp)
            cv2.imshow(self.window_name, temp)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                # Зона должна состоять минимум из трёх точек
                if len(self.current_zone) > 2:
                    self.zones.append(self.current_zone.copy())
                    self.save_zones()
                    print("Зона сохранена!")
                    self.current_zone.clear()
                else:
                    print("Недостаточно точек для зоны!")
            elif key == ord("q"):
                break

        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        """Добавляем точку при клике ЛКМ."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_zone.append((x, y))
            print(f"Добавлена точка: ({x}, {y})")

    def save_zones(self):
        """Сохраняем все зоны в JSON-файл."""
        with open(self.zones_path, "w", encoding="utf-8") as f:
            json.dump(self.zones, f, indent=4, ensure_ascii=False)
