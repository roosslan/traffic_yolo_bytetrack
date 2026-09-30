"""Загрузка и проверка конфигурации (TOML)."""

import dataclasses
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

# Пароль ClickHouse лучше не хранить в config.toml: его можно задать переменной окружения
# или строкой в файле .env рядом с config.toml. Порядок: окружение, .env, config.toml.
PASSWORD_ENV = "TRAFFIC_CLICKHOUSE_PASSWORD"
DOTENV_NAME = ".env"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Списки координат: у них элементы - списки чисел, а не строки.
_NUMBER_LISTS = {("lines", "points"), ("roi", "polygon"), ("roi", "ignore")}

_TYPE_NAMES = {
    int: "целым числом",
    float: "числом",
    str: "строкой",
    bool: "true/false",
    list: "списком",
}


class ConfigError(Exception):
    """Файл конфигурации не найден или содержит ошибку."""


@dataclass(frozen=True)
class CameraConfig:
    name: str = "cam1"  # имя камеры: попадает во все таблицы ClickHouse
    rtsp_url: str = ""
    transport: str = "tcp"  # "tcp" надёжнее, "udp" даёт чуть меньшую задержку
    reconnect_delay_sec: float = 3.0
    timeout_sec: float = 5.0  # таймаут открытия/чтения, зависший поток вызовет переподключение


@dataclass(frozen=True)
class ModelConfig:
    weights: str = "models/yolov8n.pt"  # если файла нет, ultralytics скачает его сам
    device: str = "cpu"  # "cpu", "cuda", "cuda:0"
    imgsz: int = 640  # размер, до которого сжимается кадр; больше = видны мелкие цели, но медленнее
    conf: float = 0.35  # порог уверенности детектора
    iou: float = 0.5  # порог NMS
    classes: list[str] = field(
        default_factory=lambda: ["car", "motorcycle", "bus", "truck"]
    )  # пустой список = все классы
    half: bool = False  # FP16, имеет смысл только на GPU
    tracker: str = "bytetrack.yaml"  # "bytetrack.yaml", "botsort.yaml" или путь к своему файлу


@dataclass(frozen=True)
class DayNightConfig:
    # "auto" - выбирать по картинке, "day" - всегда YOLO, "night" - всегда фары
    mode: str = "auto"
    check_interval_sec: float = 5.0  # как часто оценивать картинку в режиме auto
    switch_after_sec: float = 30.0  # новый режим включается, если держится столько секунд
    # ночь, если кадр почти серый (ИК-режим камеры) или тёмный:
    color_threshold: float = 4.0  # средняя разница каналов цвета ниже этого - кадр серый
    dark_threshold: float = 40.0  # средняя яркость ниже этого - темно


@dataclass(frozen=True)
class HeadlightsConfig:
    # кадр шире этого уменьшается до этой ширины; все размеры ниже - в пикселях такого кадра,
    # поэтому они не зависят от того, какой поток камеры обрабатывается
    work_width: int = 720
    min_brightness: int = 180  # пиксель фары ярче этого (0..255)
    min_diff: int = 50  # и ярче фона хотя бы на столько
    min_area: int = 3  # пятна меньше стольких пикселей - шум
    max_area: int = 3000  # цели больше стольких пикселей (по рамке) - засветка, а не фары
    pair_distance: float = 25.0  # пятна ближе стольких пикселей - одна машина
    background_alpha: float = 0.05  # скорость обновления фона (0..1)
    max_match_distance: float = 60.0  # дальше этого (px) от прогноза пятно - уже не та цель
    max_missed_frames: int = 10  # цель забывается, если не видна столько кадров подряд
    # проезд - повтор, если рядом (ближе стольких px) едет только что засчитанная цель:
    # фары и габариты одной машины - два пятна друг за другом; 0 - не склеивать
    merge_distance: float = 100.0


@dataclass(frozen=True)
class LineConfig:
    name: str = ""  # имя линии: попадает в таблицу crossings, должно быть уникальным
    # две точки [x, y] в долях кадра (0..1)
    points: list = field(default_factory=list)
    # названия направлений: [0] - переход с одной стороны линии на другую, [1] - обратно
    directions: list[str] = field(default_factory=lambda: ["forward", "backward"])


@dataclass(frozen=True)
class RoiConfig:
    polygon: list = field(default_factory=list)  # зона интереса [[x, y], ...]; пусто - весь кадр
    ignore: list = field(default_factory=list)  # прямоугольники [x1, y1, x2, y2], не учитываются


@dataclass(frozen=True)
class TrackingConfig:
    confirm_hits: int = 3  # цель считается захваченной после стольких подряд подтверждённых кадров
    lost_timeout_sec: float = 3.0  # цель потеряна, если её не видно дольше этого времени
    points_interval_sec: float = 0.5  # как часто писать точку траектории (0 = на каждом кадре)
    trail_length: int = 40  # длина «хвоста» траектории в окне
    reset_gap_sec: float = 5.0  # разрыв между кадрами больше этого сбрасывает трекер
    stats_interval_sec: float = 30.0  # как часто писать в журнал статистику (0 = не писать)


@dataclass(frozen=True)
class SnapshotsConfig:
    enabled: bool = True
    directory: str = "capture"
    annotate: bool = True  # рисовать рамки и номера целей на снимке


@dataclass(frozen=True)
class ClickHouseConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8124  # HTTP-порт (8124: чтобы не мешать другому ClickHouse на 8123)
    username: str = "default"
    password: str = ""
    database: str = "traffic"
    secure: bool = False  # HTTPS
    create_schema: bool = True  # при старте создать базу и таблицы, если их нет
    batch_size: int = 500  # писать, когда накопилось столько строк
    flush_interval_sec: float = 2.0  # ...или когда прошло столько секунд
    max_buffer_rows: int = 100_000  # если ClickHouse недоступен, копим не больше стольких строк
    connect_timeout_sec: float = 5.0


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    daynight: DayNightConfig = field(default_factory=DayNightConfig)
    headlights: HeadlightsConfig = field(default_factory=HeadlightsConfig)
    lines: list[LineConfig] = field(default_factory=list)  # [[lines]]: линии подсчёта
    roi: RoiConfig = field(default_factory=RoiConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    snapshots: SnapshotsConfig = field(default_factory=SnapshotsConfig)
    clickhouse: ClickHouseConfig = field(default_factory=ClickHouseConfig)


def redact_url(url: str) -> str:
    """Прячет логин и пароль в адресе, чтобы адрес можно было безопасно писать в журнал:
    rtsp://admin:secret@10.0.0.5/1 -> rtsp://***@10.0.0.5/1. Адрес без учётных данных
    возвращается без изменений.

    Args:
        url: адрес, обычно RTSP-адрес камеры.

    Returns:
        адрес, в котором всё до «@» в части с хостом заменено на «***».
    """
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=f"***@{host}"))


def load_config(path: str | Path) -> Config:
    """Читает файл конфигурации TOML, проверяет его (см. parse_config) и подставляет пароль
    ClickHouse из внешних источников. Пароль берётся из первого источника, где он задан:
    переменная окружения TRAFFIC_CLICKHOUSE_PASSWORD, затем строка с тем же именем в файле .env
    рядом с файлом конфигурации, затем значение password из самого config.toml.

    Args:
        path: путь к файлу конфигурации (строка или Path).

    Returns:
        готовый объект Config.

    Raises:
        ConfigError: файла нет, в нём ошибка синтаксиса TOML, неизвестный раздел или ключ,
            значение неверного типа или вне допустимого диапазона, либо .env не читается.
    """
    path = Path(path)
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(
            f"Файл конфигурации '{path}' не найден. "
            f"Скопируйте config.example.toml в {path} и отредактируйте."
        ) from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Некорректный TOML в '{path}': {exc}") from exc
    config = parse_config(raw)
    password = os.environ.get(PASSWORD_ENV) or read_dotenv(path.parent / DOTENV_NAME).get(
        PASSWORD_ENV
    )
    if password:
        config = dataclasses.replace(
            config, clickhouse=dataclasses.replace(config.clickhouse, password=password)
        )
    return config


def read_dotenv(path: Path) -> dict[str, str]:
    """Читает простой файл .env со строками вида КЛЮЧ=значение. Пустые строки, комментарии (#...)
    и строки без «=» пропускаются. Пробелы вокруг ключа и значения обрезаются, одна пара
    одинаковых кавычек вокруг значения ("..." или '...') снимается. Метка BOM в начале файла
    (её добавляет Блокнот Windows) не мешает. Подстановка переменных и многострочные значения
    не поддерживаются.

    Args:
        path: путь к файлу .env.

    Returns:
        словарь {ключ: значение}; пустой, если файла нет.

    Raises:
        ConfigError: файл есть, но прочитать его не удалось.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"Не удалось прочитать '{path}': {exc}") from exc
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def parse_config(raw: dict[str, Any]) -> Config:
    """Превращает уже разобранный TOML (словарь) в объект Config. Каждый раздел ([camera],
    [model], [daynight], [headlights], [roi], [tracking], [snapshots], [clickhouse])
    собирается в свой класс настроек, каждая таблица [[lines]] - в LineConfig; отсутствующие
    разделы и ключи получают значения по умолчанию. Затем все значения проверяются на
    допустимость (см. _validate).

    Args:
        raw: словарь из tomllib.load(): {имя раздела: {ключ: значение}}, для lines - список
            таких словарей.

    Returns:
        проверенный Config.

    Raises:
        ConfigError: неизвестный раздел или ключ, неверный тип или недопустимое значение.
    """
    sections = {f.name: f for f in dataclasses.fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ConfigError(f"Неизвестные разделы конфигурации: {', '.join(sorted(unknown))}")

    lines = raw.get("lines", [])
    if not isinstance(lines, list):
        raise ConfigError("lines должен быть массивом таблиц [[lines]]")
    built = {
        name: _build_section(sections[name].default_factory, name, raw.get(name, {}))
        for name in sections
        if name != "lines"
    }
    built["lines"] = [_build_section(LineConfig, "lines", item) for item in lines]
    config = Config(**built)
    _validate(config)
    return config


def _build_section(cls: type, name: str, data: Any) -> Any:
    """Собирает один раздел конфигурации в объект класса настроек и проверяет типы значений.
    Ожидаемый тип каждого ключа берётся из его значения по умолчанию в классе. Целое число
    принимается там, где ожидается дробное (conf = 1 вместо 1.0), но true/false вместо числа
    не принимается. У списков дополнительно проверяется, что все элементы - строки.

    Args:
        cls: класс настроек раздела (CameraConfig, ModelConfig и т. д.).
        name: имя раздела, как в TOML, например "camera"; нужно только для текста ошибок.
        data: содержимое раздела из TOML: {ключ: значение}.

    Returns:
        экземпляр cls; ключи, которых нет в data, получают значения по умолчанию.

    Raises:
        ConfigError: раздел не является таблицей, в нём есть неизвестный ключ или значение
            неверного типа.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] должен быть таблицей")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Неизвестные ключи в [{name}]: {', '.join(sorted(unknown))}")
    for key, value in data.items():
        expected = type(getattr(cls(), key))
        # целое число подходит там, где ждут дробное (например, `conf = 1`)
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            continue
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ConfigError(
                f"[{name}] {key} должен быть {_TYPE_NAMES[expected]}, "
                f"а сейчас {type(value).__name__}"
            )
        if expected is list and (name, key) in _NUMBER_LISTS:
            if not all(isinstance(it, list) and all(_is_number(v) for v in it) for it in value):
                raise ConfigError(f"[{name}] {key} должен быть списком списков чисел")
        elif expected is list and not all(isinstance(item, str) for item in value):
            raise ConfigError(f"[{name}] {key} должен быть списком строк")
    return cls(**data)


def _is_number(value: Any) -> bool:
    """Число ли это (целое или дробное), но не true/false.

    Args:
        value: проверяемое значение.

    Returns:
        True для int и float, кроме bool.
    """
    return isinstance(value, int | float) and not isinstance(value, bool)


def _fractions(points: list, size: int) -> bool:
    """Все ли элементы - списки из size чисел в диапазоне 0..1 (координаты в долях кадра).

    Args:
        points: список списков чисел.
        size: сколько чисел должно быть в каждом элементе (2 для точки, 4 для прямоугольника).

    Returns:
        True, если все элементы подходят.
    """
    return all(len(p) == size and all(0 <= v <= 1 for v in p) for p in points)


def _validate(config: Config) -> None:
    """Проверяет, что значения всей конфигурации допустимы:
      - обязательные строки не пустые (имя камеры, адрес потока, путь к весам и т. п.);
      - числа в разумных пределах (порог уверенности в (0, 1), порт 1..65535, таймауты
        положительные и т. д.);
      - транспорт только "tcp" или "udp";
      - имя базы - допустимый идентификатор ClickHouse (оно подставляется прямо в SQL);
      - лимит буфера не меньше размера пакета.
    Останавливается на первой найденной ошибке.

    Args:
        config: собранная конфигурация, которую нужно проверить.

    Raises:
        ConfigError: найдено недопустимое значение; в тексте описана первая ошибка.
    """
    cam, mdl, trk = config.camera, config.model, config.tracking
    dn, hl, roi = config.daynight, config.headlights, config.roi
    snap, ch = config.snapshots, config.clickhouse
    _require(bool(cam.name), "[camera] name не должен быть пустым")
    _require(bool(cam.rtsp_url), "[camera] rtsp_url обязателен")
    _require(cam.transport in ("tcp", "udp"), "[camera] transport должен быть 'tcp' или 'udp'")
    _require(cam.reconnect_delay_sec >= 0, "[camera] reconnect_delay_sec должен быть >= 0")
    _require(cam.timeout_sec > 0, "[camera] timeout_sec должен быть > 0")

    _require(bool(mdl.weights), "[model] weights не должен быть пустым")
    _require(bool(mdl.device), "[model] device не должен быть пустым")
    _require(32 <= mdl.imgsz <= 4096, "[model] imgsz должен быть в диапазоне 32..4096")
    _require(0 < mdl.conf < 1, "[model] conf должен быть в диапазоне (0, 1)")
    _require(0 < mdl.iou <= 1, "[model] iou должен быть в диапазоне (0, 1]")
    _require(bool(mdl.tracker), "[model] tracker не должен быть пустым")

    _require(
        dn.mode in ("auto", "day", "night"), "[daynight] mode должен быть 'auto', 'day' или 'night'"
    )
    _require(dn.check_interval_sec > 0, "[daynight] check_interval_sec должен быть > 0")
    _require(dn.switch_after_sec >= 0, "[daynight] switch_after_sec должен быть >= 0")
    _require(dn.color_threshold >= 0, "[daynight] color_threshold должен быть >= 0")
    _require(0 <= dn.dark_threshold <= 255, "[daynight] dark_threshold должен быть 0..255")

    _require(hl.work_width >= 32, "[headlights] work_width должен быть >= 32")
    _require(0 <= hl.min_brightness <= 255, "[headlights] min_brightness должен быть 0..255")
    _require(0 <= hl.min_diff <= 255, "[headlights] min_diff должен быть 0..255")
    _require(hl.min_area >= 1, "[headlights] min_area должен быть >= 1")
    _require(hl.max_area > hl.min_area, "[headlights] max_area должен быть больше min_area")
    _require(hl.pair_distance >= 0, "[headlights] pair_distance должен быть >= 0")
    _require(0 < hl.background_alpha <= 1, "[headlights] background_alpha должен быть в (0, 1]")
    _require(hl.max_match_distance > 0, "[headlights] max_match_distance должен быть > 0")
    _require(hl.max_missed_frames >= 0, "[headlights] max_missed_frames должен быть >= 0")
    _require(hl.merge_distance >= 0, "[headlights] merge_distance должен быть >= 0")

    names = [line.name for line in config.lines]
    _require(all(names), "[[lines]] name: у каждой линии должно быть имя")
    _require(len(set(names)) == len(names), "[[lines]] name: имена линий должны различаться")
    for line in config.lines:
        where = f"[[lines]] '{line.name}'"
        _require(
            len(line.points) == 2 and _fractions(line.points, 2),
            f"{where} points: нужны две точки [x, y] с координатами 0..1",
        )
        _require(line.points[0] != line.points[1], f"{where} points: начало и конец совпадают")
        _require(
            len(line.directions) == 2
            and all(line.directions)
            and line.directions[0] != line.directions[1],
            f"{where} directions: нужны два разных непустых названия направлений",
        )
    _require(
        not roi.polygon or (len(roi.polygon) >= 3 and _fractions(roi.polygon, 2)),
        "[roi] polygon: нужно не меньше трёх точек [x, y] с координатами 0..1 (или пустой список)",
    )
    _require(
        _fractions(roi.ignore, 4)
        and all(x1 < x2 and y1 < y2 for x1, y1, x2, y2 in roi.ignore),
        "[roi] ignore: прямоугольники [x1, y1, x2, y2] с координатами 0..1, x1 < x2, y1 < y2",
    )

    _require(trk.confirm_hits >= 1, "[tracking] confirm_hits должен быть >= 1")
    _require(trk.lost_timeout_sec > 0, "[tracking] lost_timeout_sec должен быть > 0")
    _require(trk.points_interval_sec >= 0, "[tracking] points_interval_sec должен быть >= 0")
    _require(trk.trail_length >= 1, "[tracking] trail_length должен быть >= 1")
    _require(trk.reset_gap_sec > 0, "[tracking] reset_gap_sec должен быть > 0")
    _require(trk.stats_interval_sec >= 0, "[tracking] stats_interval_sec должен быть >= 0")

    _require(bool(snap.directory), "[snapshots] directory не должен быть пустым")

    _require(bool(ch.host), "[clickhouse] host не должен быть пустым")
    _require(1 <= ch.port <= 65535, "[clickhouse] port должен быть в диапазоне 1..65535")
    _require(
        bool(_IDENTIFIER.match(ch.database)),
        "[clickhouse] database может состоять только из латинских букв, цифр и '_'",
    )
    _require(ch.batch_size >= 1, "[clickhouse] batch_size должен быть >= 1")
    _require(ch.flush_interval_sec > 0, "[clickhouse] flush_interval_sec должен быть > 0")
    _require(
        ch.max_buffer_rows >= ch.batch_size,
        "[clickhouse] max_buffer_rows не может быть меньше batch_size",
    )
    _require(ch.connect_timeout_sec > 0, "[clickhouse] connect_timeout_sec должен быть > 0")


def _require(condition: bool, message: str) -> None:
    """Вспомогательная проверка для _validate: если условие ложно, выбрасывает ConfigError.

    Args:
        condition: условие, которое должно выполняться.
        message: текст ошибки для пользователя: что не так и в каком разделе.

    Raises:
        ConfigError: условие ложно; текст ошибки - message.
    """
    if not condition:
        raise ConfigError(message)
