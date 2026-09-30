"""Командная строка: RTSP -> день: YOLO + ByteTrack / ночь: фары -> линия подсчёта -> ClickHouse."""

import argparse
import importlib.metadata
import logging
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from traffic import storage
from traffic.capture import LatestFrameReader, make_rtsp_capture
from traffic.config import Config, ConfigError, load_config, redact_url
from traffic.counter import Zone
from traffic.daynight import DayNightTracker
from traffic.detector import YoloTracker
from traffic.headlights import CLASS_NAME as HEADLIGHTS
from traffic.headlights import HeadlightsTracker
from traffic.overlay import annotated, draw_status, draw_tracks, draw_zone
from traffic.pipeline import TrackingWorker
from traffic.recorder import TargetRecorder
from traffic.snapshots import SnapshotWriter
from traffic.tracks import TrackRegistry

log = logging.getLogger("traffic")

WINDOW = "traffic_yolo_bytetrack"
NO_SIGNAL_AFTER_SEC = 3.0


def build_parser() -> argparse.ArgumentParser:
    """Создаёт разбор аргументов командной строки. Описывает ключи:
      --config PATH  - путь к файлу конфигурации TOML (по умолчанию config.toml);
      --headless     - работать без окна с видео (для сервера или службы);
      --video PATH   - обработать видеофайл вместо камеры (для настройки линии и порогов);
      --self-test    - проверить, что библиотеки и модель загружаются, и выйти;
      --init-db      - создать базу и таблицы в ClickHouse и выйти;
      --list [N]     - показать последние N проездов из ClickHouse (по умолчанию 20) и выйти;
      --counts [H]   - показать число проездов по часам за последние H часов (по умолчанию 24).

    Returns:
        готовый argparse.ArgumentParser; сам разбор аргументов делает вызывающий код.
    """
    parser = argparse.ArgumentParser(prog="traffic", description=__doc__)
    parser.add_argument("--config", default="config.toml", help="путь к конфигурации (TOML)")
    parser.add_argument("--headless", action="store_true", help="без окна с видео")
    parser.add_argument(
        "--video",
        metavar="PATH",
        help="обработать видеофайл вместо камеры (ClickHouse не используется)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить, что PyTorch, ultralytics и модель загружаются, и выйти",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="создать базу и таблицы в ClickHouse и выйти",
    )
    parser.add_argument(
        "--list",
        nargs="?",
        const=20,
        type=int,
        metavar="N",
        help="показать последние N проездов из ClickHouse (по умолчанию 20) и выйти",
    )
    parser.add_argument(
        "--counts",
        nargs="?",
        const=24,
        type=int,
        metavar="HOURS",
        help="показать число проездов по часам за последние HOURS часов (по умолчанию 24)",
    )
    return parser


def run_self_test(config: Config) -> None:
    """Самопроверка установки. Импортирует OpenCV, numpy, PyTorch и ultralytics, пишет в журнал
    их версии (и версию clickhouse-connect) и доступна ли CUDA. Затем прогоняет по чёрному
    кадру 640x480 и модель YOLO (вместе с трекером), и ночной детектор фар. Любая проблема
    (нет библиотеки, нет CUDA при device = "cuda", испорченные веса) выбрасывается как
    исключение.

    Args:
        config: конфигурация; используются разделы [model], [headlights], [[lines]] и [roi].

    Returns:
        None. Успехом считается отсутствие исключения.
    """
    import cv2
    import numpy
    import torch
    import ultralytics

    log.info(
        "torch %s, ultralytics %s, opencv %s, clickhouse-connect %s",
        torch.__version__,
        ultralytics.__version__,
        cv2.__version__,
        importlib.metadata.version("clickhouse-connect"),
    )
    log.info("CUDA доступна: %s", torch.cuda.is_available())
    frame = numpy.zeros((480, 640, 3), numpy.uint8)
    detector = YoloTracker(config.model)
    detector.load()
    detector.track(frame)
    headlights = HeadlightsTracker(config.headlights, build_zone(config))
    headlights.track(frame)  # первый кадр только запоминает фон
    headlights.track(frame)


def main(argv: list[str] | None = None) -> int:
    """Точка входа программы. Разбирает аргументы, настраивает журнал (уровень INFO, формат
    «время уровень сообщение») и выбирает режим работы:
      --self-test                   - самопроверка; config.toml не обязателен, без него
                                      проверяются настройки по умолчанию;
      --init-db / --list / --counts - одна служебная команда к ClickHouse (см. database_command);
      --video PATH                  - обработка видеофайла (см. run_video);
      без этих ключей               - основной режим: подсчёт машин с камеры (см. run).

    Args:
        argv: аргументы командной строки без имени программы. None (по умолчанию) означает «взять из
            sys.argv»; явный список удобен в тестах.

    Returns:
        код завершения процесса: 0 - успех; 1 - ошибка работы (самопроверка не пройдена, модель не
        загрузилась, ClickHouse недоступен, видео не открылось); 2 - ошибка в конфигурации.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    config_path = Path(args.config)

    if args.self_test:
        # Конфигурация здесь необязательна: без неё проверяются настройки по умолчанию.
        try:
            config = load_config(config_path) if config_path.exists() else Config()
            run_self_test(config)
        except Exception:
            log.exception("Самопроверка не пройдена")
            return 1
        log.info("Самопроверка пройдена")
        return 0

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    if args.init_db or args.list is not None or args.counts is not None:
        return database_command(config, init=args.init_db, count=args.list, hours=args.counts)
    if args.video:
        return run_video(config, Path(args.video), headless=args.headless)
    return run(config, headless=args.headless)


def database_command(
    config: Config, init: bool, count: int | None, hours: int | None = None
) -> int:
    """Выполняет одну служебную команду к ClickHouse и завершает работу:
      init           - создать базу и таблицы по schema.sql (когда их ещё нет);
      hours не None  - напечатать число проездов по часам, линиям и направлениям за hours
                       часов;
      иначе          - напечатать последние проезды, новые сверху.
    Время печатается местное.

    Args:
        config: полная конфигурация; используется только раздел [clickhouse] (адрес, учётные данные,
            имя базы).
        init: True: создать схему (ключ --init-db).
        count: сколько последних проездов показать (ключ --list); None и 0 заменяются на 20.
        hours: за сколько часов считать проезды (ключ --counts); None - не считать, 0
            заменяется на 24.

    Returns:
        0 при успехе, 1 если ClickHouse недоступен или запрос завершился ошибкой.
    """
    ch = config.clickhouse
    try:
        client = storage.default_client_factory(ch)
        if init:
            storage.ensure_schema(client, ch.database)
            print(f"База '{ch.database}' и таблицы готовы ({ch.host}:{ch.port})")
            return 0
        if hours is not None:
            rows = storage.hourly_counts(client, ch.database, hours or 24)
        else:
            rows = storage.recent_crossings(client, ch.database, count or 20)
    except Exception as exc:
        log.error("ClickHouse недоступен (%s:%s): %s", ch.host, ch.port, exc)
        return 1
    if not rows:
        print(f"В {ch.database}.crossings пока нет записей")
    if hours is not None:
        for hour, camera, line, direction, cars in rows:
            print(f"{local(hour):%Y-%m-%d %H:00}  {camera}  {line:12} {direction:12} {cars}")
        return 0
    for ts, camera, line, direction, class_name, track_id, conf, snapshot in rows:
        when = f"{local(ts):%Y-%m-%d %H:%M:%S}"
        print(
            f"{when}  {camera}  {line:12} {direction:12} {class_name} #{track_id} {conf:.2f}"
            f"  {snapshot}"
        )
    return 0


def local(moment: datetime) -> datetime:
    """Переводит время из ClickHouse (там всё в UTC) в местное время компьютера для вывода.
    Время без часового пояса считается временем UTC.

    Args:
        moment: время из ClickHouse.

    Returns:
        то же время в местном часовом поясе.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone()


def build_zone(config: Config) -> Zone:
    """Собирает место подсчёта из таблиц [[lines]] и раздела [roi].

    Args:
        config: полная конфигурация.

    Returns:
        Zone; без линий подсчёта проезды не считаются.
    """
    return Zone(
        [(line.name, line.points, line.directions) for line in config.lines],
        roi=config.roi.polygon,
        ignore=config.roi.ignore,
    )


def build_pipeline(
    config: Config,
    sink: storage.ClickHouseSink | None,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
) -> tuple[DayNightTracker, TrackRegistry, TargetRecorder, Zone]:
    """Собирает общую для камеры и видеофайла часть: детектор день/ночь, место подсчёта,
    реестр целей и обработчик событий (снимки, журнал, ClickHouse).

    Args:
        config: полная конфигурация.
        sink: запись в ClickHouse или None.
        clock: монотонные часы для реестра и выбора режима; для видеофайла - время внутри видео.
        wall_clock: системные часы для меток времени.

    Returns:
        (детектор, реестр целей, обработчик событий, место подсчёта).
    """
    snaps, trk, hl = config.snapshots, config.tracking, config.headlights
    zone = build_zone(config)
    if not config.lines:
        log.warning("Линии подсчёта не заданы ([[lines]]): проезды не считаются")
    recorder = TargetRecorder(
        camera=config.camera.name,
        session_id=uuid.uuid4(),
        sink=sink,
        snapshots=SnapshotWriter(snaps.directory) if snaps.enabled else None,
        annotate=(lambda frame, views: annotated(frame, views, zone)) if snaps.annotate else None,
    )
    registry = TrackRegistry(
        on_captured=recorder.on_captured,
        on_point=recorder.on_point,
        on_lost=recorder.on_lost,
        on_crossed=recorder.on_crossed,
        zone=zone,
        # в конфигурации - пиксели рабочего кадра ночного детектора, реестру нужна доля ширины
        merge_distance={HEADLIGHTS: hl.merge_distance / hl.work_width},
        confirm_hits=trk.confirm_hits,
        lost_timeout=trk.lost_timeout_sec,
        points_interval=trk.points_interval_sec,
        trail_length=trk.trail_length,
        clock=clock,
        wall_clock=wall_clock,
    )
    detector = DayNightTracker(
        config.daynight,
        day=YoloTracker(config.model),
        night=HeadlightsTracker(config.headlights, zone),
        # у дневного и ночного детекторов своя нумерация целей: старые цели закрываем
        on_switch=lambda mode: registry.close_all("mode"),
        clock=clock,
    )
    return detector, registry, recorder, zone


def run(config: Config, headless: bool) -> int:
    """Основной режим работы. Загружает модели, запускает запись в ClickHouse (если она включена)
    и собирает цепочку обработки:
      чтение RTSP-потока (LatestFrameReader)
      -> детектор день/ночь в фоновом потоке (TrackingWorker)
      -> реестр целей с линией подсчёта (TrackRegistry)
      -> обработчик событий (TargetRecorder: снимки, журнал, ClickHouse).
    Затем ждёт, пока пользователь закроет окно, нажмёт q или Ctrl+C. При выходе останавливает
    поток обработки, закрывает все сопровождаемые цели с причиной 'shutdown', дописывает
    остаток буфера в ClickHouse, отключается от камеры и пишет в журнал итог проездов.

    Args:
        config: полная загруженная и проверенная конфигурация (все разделы).
        headless: True: работать без окна, остановка только по Ctrl+C; False: показывать окно с
            видео, целями, линией и счётчиками.

    Returns:
        0 после штатной остановки, 1 если не удалось загрузить модель.
    """
    ch, cam = config.clickhouse, config.camera
    sink = storage.ClickHouseSink(ch) if ch.enabled else None
    detector, registry, recorder, zone = build_pipeline(config, sink)
    try:
        started = time.monotonic()
        detector.load()
        log.info("Модели загружены за %.1f с", time.monotonic() - started)
    except Exception:
        log.exception("Не удалось загрузить модель")
        return 1

    if sink is not None:
        sink.start()
        log.info("ClickHouse: %s:%s, база '%s'", ch.host, ch.port, ch.database)
    else:
        log.info("Запись в ClickHouse отключена, проезды идут только в журнал")

    log.info("Камера '%s': %s (%s)", cam.name, redact_url(cam.rtsp_url), cam.transport)
    reader = LatestFrameReader(
        lambda: make_rtsp_capture(cam.rtsp_url, cam.transport, cam.timeout_sec),
        cam.reconnect_delay_sec,
    )
    worker = TrackingWorker(
        reader,
        detector,
        registry,
        reset_gap=config.tracking.reset_gap_sec,
        stats_interval=config.tracking.stats_interval_sec,
    )

    reader.start()
    worker.start()
    try:
        if headless:
            run_headless()
        else:
            run_viewer(
                reader,
                registry,
                lambda canvas: draw_overlay(canvas, config, detector, recorder, zone),
            )
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        registry.close_all("shutdown")  # итог по целям, которые ещё сопровождались
        if sink is not None:
            sink.stop()
        reader.stop()
        log.info("Итого проездов: %s", format_counts(recorder.counts(), config))
    return 0


def run_video(config: Config, path: Path, headless: bool) -> int:
    """Обрабатывает видеофайл кадр за кадром, без пропусков, и пишет итог в журнал. Время
    берётся из самого видео (номер кадра / частота кадров), поэтому результат не зависит от
    скорости компьютера. ClickHouse не используется; снимки пишутся, как настроено
    в [snapshots]. Режим нужен, чтобы подобрать линию подсчёта и пороги на записи, а не на
    живой камере.

    Args:
        config: полная конфигурация.
        path: путь к видеофайлу.
        headless: True - без окна (быстрее); False - показывать кадры с разметкой, q - выход.

    Returns:
        0 после обработки файла, 1 если файл не открылся или модель не загрузилась.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        log.error("Не удалось открыть видео '%s'", path)
        return 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not 0 < fps < 1000:
        fps = 20.0  # камеры иногда пишут в файл нелепую частоту кадров
    video_time = [0.0]
    started_wall = time.time()
    detector, registry, recorder, zone = build_pipeline(
        config,
        sink=None,
        clock=lambda: video_time[0],
        wall_clock=lambda: started_wall + video_time[0],
    )
    try:
        detector.load()
    except Exception:
        log.exception("Не удалось загрузить модель")
        capture.release()
        return 1

    log.info("Видео '%s', %.1f кадр/с", path, fps)
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            video_time[0] = index / fps
            index += 1
            registry.update(detector.track(frame), frame, now=video_time[0])
            if not headless:
                canvas = frame.copy()
                draw_overlay(canvas, config, detector, recorder, zone)
                draw_tracks(canvas, registry.views())
                cv2.imshow(WINDOW, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        registry.close_all("shutdown")
        if not headless:
            cv2.destroyAllWindows()
    log.info(
        "Обработано %d кадров (%.0f с видео). Проездов: %s",
        index,
        index / fps,
        format_counts(recorder.counts(), config),
    )
    return 0


def format_counts(counts: dict[tuple[str, str], int], config: Config) -> str:
    """Форматирует счётчики для журнала: по линиям и направлениям в порядке конфигурации.

    Args:
        counts: {(линия, направление): число проездов}.
        config: конфигурация; из неё берутся линии и названия направлений.

    Returns:
        строка вида «west: forward 5, backward 3; east: forward 4, backward 2».
    """
    parts = []
    for line in config.lines:
        numbers = ", ".join(f"{d} {counts.get((line.name, d), 0)}" for d in line.directions)
        parts.append(f"{line.name}: {numbers}")
    return "; ".join(parts) or "линий нет"


def draw_overlay(
    canvas: Any, config: Config, detector: DayNightTracker, recorder: TargetRecorder, zone: Zone
) -> None:
    """Рисует на кадре место подсчёта, режим и счётчики (цели рисуются отдельно).

    Args:
        canvas: изображение (массив numpy BGR); изменяется на месте.
        config: конфигурация; из неё берутся линии и названия направлений.
        detector: детектор день/ночь; из него берётся текущий режим.
        recorder: обработчик событий; из него берутся счётчики.
        zone: место подсчёта.
    """
    draw_zone(canvas, zone)
    lines = [(line.name, list(line.directions)) for line in config.lines]
    draw_status(canvas, detector.mode, recorder.counts(), lines)


def run_headless() -> None:
    """Режим без окна: держит главный поток живым, пока вся работа идёт в фоновых потоках.
    Цикл бесконечный; прерывается только исключением KeyboardInterrupt (Ctrl+C), которое
    перехватывает вызывающая функция run.

    Returns:
        управление не возвращает, выход только через исключение.
    """
    log.info("Работаю без окна, остановка: Ctrl+C")
    while True:
        time.sleep(1.0)


def run_viewer(
    reader: LatestFrameReader,
    registry: TrackRegistry,
    decorate: Callable[[Any], None] | None = None,
) -> None:
    """Режим с окном: показывает свежие кадры с камеры с нарисованными целями (рамки, номера,
    «хвосты» траекторий), линией подсчёта и счётчиками. Рисует на копии кадра, потому что тот
    же кадр в это время читает поток детекции. Если новых кадров нет дольше
    NO_SIGNAL_AFTER_SEC секунд, поверх последнего кадра пишется красное «NO SIGNAL». Выходит
    по клавише q или когда окно закрыто крестиком; при выходе закрывает все окна OpenCV.

    Args:
        reader: источник кадров: из него берутся самый свежий кадр и время его получения.
        registry: реестр целей: из него берётся текущее состояние целей для рисования.
        decorate: функция (кадр) -> None, дорисовывающая линию, режим и счётчики; None - не
            рисовать.

    Returns:
        None, когда пользователь закрыл окно или нажал q.
    """
    import cv2

    last_seq = 0
    try:
        while True:
            frame = reader.wait_for_new(last_seq, timeout=0.1) or reader.latest()
            if frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
            last_seq = frame.seq
            # кадр читает и поток сопровождения, поэтому рисуем на копии
            canvas = frame.image.copy()
            if decorate is not None:
                decorate(canvas)
            draw_tracks(canvas, registry.views())
            if time.monotonic() - frame.timestamp > NO_SIGNAL_AFTER_SEC:
                cv2.putText(
                    canvas, "NO SIGNAL", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 2
                )
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # окно закрыли крестиком
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
