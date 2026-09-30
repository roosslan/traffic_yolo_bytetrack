"""Рисование целей на кадре: рамки, номера, «хвосты» траекторий."""

import colorsys
from typing import Any

from traffic.counter import Zone
from traffic.tracks import TrackView


def track_color(track_id: int) -> tuple[int, int, int]:
    """Даёт устойчивый цвет для номера цели: одна и та же цель всегда рисуется одним цветом,
    а соседние номера получают заметно разные цвета (оттенок сдвигается на золотое сечение).

    Args:
        track_id: номер цели от трекера.

    Returns:
        цвет (синий, зелёный, красный), каждая составляющая 0..255, как ждёт OpenCV.
    """
    hue = (track_id * 0.61803398875) % 1.0  # золотое сечение: соседние номера сильно различаются
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(canvas: Any, views: list[TrackView]) -> None:
    """Рисует цели прямо на переданном изображении (изменяет его): рамку, линию траектории
    («хвост» из последних центров рамки) и подпись «класс #номер уверенность» на цветной
    плашке над рамкой. Захваченные цели обводятся толстой рамкой, ещё не подтверждённые -
    тонкой. Цели, чей проезд уже засчитан, рисуются зелёным.

    Args:
        canvas: изображение (массив numpy BGR), на котором рисовать; изменяется на месте.
        views: состояния целей: рамка, номер, класс, уверенность, траектория и признак захвата.
    """
    import cv2

    for view in views:
        color = (0, 255, 0) if view.counted else track_color(view.track_id)
        x1, y1, x2, y2 = (int(v) for v in view.box)
        thickness = 3 if view.captured else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        if len(view.trail) > 1:
            points = [(int(x), int(y)) for x, y in view.trail]
            for start, end in zip(points, points[1:], strict=False):
                cv2.line(canvas, start, end, color, 2)

        label = f"{view.class_name} #{view.track_id} {view.conf:.2f}"
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        top = max(y1 - height - 8, 0)
        cv2.rectangle(canvas, (x1, top), (x1 + width + 6, top + height + 8), color, cv2.FILLED)
        cv2.putText(
            canvas, label, (x1 + 3, top + height + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )


def annotated(frame: Any, views: list[TrackView], zone: Zone | None = None) -> Any:
    """Возвращает копию кадра с нарисованными целями (и линией подсчёта, если задана зона);
    исходный кадр не меняется. Используется для снимков в момент проезда.

    Args:
        frame: исходный кадр (массив numpy BGR).
        views: цели для рисования (см. draw_tracks).
        zone: место подсчёта для рисования (см. draw_zone); None - не рисовать.

    Returns:
        новый массив numpy с рамками, траекториями и подписями.
    """
    canvas = frame.copy()
    draw_zone(canvas, zone)
    draw_tracks(canvas, views)
    return canvas


def draw_zone(canvas: Any, zone: Zone | None) -> None:
    """Рисует на кадре зону интереса (жёлтым контуром), игнорируемые области (серым)
    и линии подсчёта (красным, с именем линии у первой точки). Изменяет canvas на месте.

    Args:
        canvas: изображение (массив numpy BGR).
        zone: зона; None - ничего не рисуется.
    """
    import cv2
    import numpy as np

    if zone is None:
        return
    height, width = canvas.shape[:2]
    zone.bind(width, height)
    if zone.roi:
        polygon = np.array([[round(x), round(y)] for x, y in zone.roi], np.int32)
        cv2.polylines(canvas, [polygon], True, (0, 220, 255), 1)
    for x1, y1, x2, y2 in zone.ignore:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1)
    for line in zone.lines:
        p1 = (round(line.p1[0]), round(line.p1[1]))
        p2 = (round(line.p2[0]), round(line.p2[1]))
        cv2.line(canvas, p1, p2, (0, 0, 255), 2)
        label = (p1[0] + 4, max(p1[1] - 6, 12))
        cv2.putText(canvas, line.name, label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(canvas, line.name, label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)


def draw_status(
    canvas: Any,
    mode: str | None,
    counts: dict[tuple[str, str], int],
    lines: list[tuple[str, list[str]]],
) -> None:
    """Пишет в правом верхнем углу режим (день/ночь) и счётчики проездов: по строке на линию,
    «имя: направление число / направление число». Правый верх выбран потому, что внизу кадра
    обычно дорога.

    Args:
        canvas: изображение (массив numpy BGR); изменяется на месте.
        mode: 'day', 'night' или None (режим ещё не выбран).
        counts: {(линия, направление): число проездов}.
        lines: линии в порядке вывода: (имя, названия двух направлений).
    """
    import cv2

    rows = [f"mode: {mode or '...'}"]
    for name, directions in lines:
        parts = [f"{d} {counts.get((name, d), 0)}" for d in directions]
        rows.append(f"{name}: {' / '.join(parts)}")
    width = canvas.shape[1]
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.55
    for i, text in enumerate(rows):
        (text_width, _), _ = cv2.getTextSize(text, font, scale, 2)
        origin = (max(width - text_width - 10, 0), 22 + i * 22)
        cv2.putText(canvas, text, origin, font, scale, (0, 0, 0), 4)
        cv2.putText(canvas, text, origin, font, scale, (0, 255, 255), 2)
