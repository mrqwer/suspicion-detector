import argparse
from app.detection.human_detector import HumanDetector
from app.zone.zone_handler import ZoneHandler
from app.alarm.alarm_handler import AlarmHandler
from app.video.video_processor import VideoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Система обнаружения вторжений в зону ограниченного доступа"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Путь к видео файлу",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="Путь к модели YOLO"
    )
    parser.add_argument(
        "--zones", type=str, default="restricted_zones.json", help="Путь к файлу зон"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Устройство для инференса (cpu/cuda)",
    )
    parser.add_argument("--no_tracking", action="store_true", help="Отключить трекинг")
    return parser.parse_args()


def main():
    args = parse_args()
    # Инициализация компонентов
    detector = HumanDetector(model_path=args.model, device=args.device)
    zone_handler = ZoneHandler(args.zones)
    alarm_handler = AlarmHandler()

    # Если файл зон не существует, предлагаем создать зоны
    if not zone_handler.zones:
        print("Файл с запретными зонами не найден. Создайте зоны...")
        zone_handler.select_zone(args.video)

    # Обработка видео
    processor = VideoProcessor(
        detector=detector, zone_handler=zone_handler, alarm_handler=alarm_handler, use_tracking=not args.no_tracking
    )

    processor.run(args.video)


if __name__ == "__main__":
    main()
