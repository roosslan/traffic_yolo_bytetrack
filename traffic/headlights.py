"""Ночной детектор: машины по свету фар.

Ночью камера почти ничего не видит, кроме фар. Поэтому ночью вместо нейросети
работает простой разбор кадра:

1. **фон**: медленно обновляемое среднее яркости кадра; уличные фонари, горящие окна и блики
   на стекле становятся частью фона и не мешают;
2. **пятна**: пиксели намного ярче фона и сами по себе яркие;
3. **машина**: пятна ближе `pair_distance` пикселей друг к другу склеиваются в одну цель
   (две фары одной машины, отражение фар на асфальте);
4. **сопровождение**: цели на соседних кадрах связываются по ближайшему центру с учётом скорости
   (`CentroidTracker`). IoU-трекер (ByteTrack) для крошечных быстро движущихся точек не подходит:
   рамки соседних кадров часто вообще не пересекаются.

Кадр шире `work_width` перед разбором уменьшается до этой ширины: так основной поток камеры
(2560 px) обрабатывается так же быстро, как дополнительный (720 px), и все пороги в пикселях
одинаково подходят к обоим. Рамки целей возвращаются в пикселях исходного кадра.

Результат - те же `Observation`, что даёт YOLO, с классом `headlights`.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from traffic.config import HeadlightsConfig
from traffic.counter import Zone
from traffic.detector import Observation

log = logging.getLogger(__name__)

CLASS_NAME = "headlights"
CLASS_ID = -1  # у ночных целей нет номера класса COCO

Box = tuple[float, float, float, float]


@dataclass
class _Blob:
    box: Box
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        """Центр рамки пятна.

        Returns:
            точка (x, y) в пикселях.
        """
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class _Target:
    track_id: int
    center: tuple[float, float]
    velocity: tuple[float, float]
    missed: int = 0  # сколько кадров подряд цель не найдена

    def predicted(self) -> tuple[float, float]:
        """Где цель ожидается на следующем кадре: текущее положение плюс скорость, умноженная на
        число кадров, прошедших с последнего наблюдения.

        Returns:
            точка (x, y) в пикселях.
        """
        steps = self.missed + 1
        return self.center[0] + self.velocity[0] * steps, self.center[1] + self.velocity[1] * steps


class CentroidTracker:
    """Связывает пятна на соседних кадрах в цели с постоянными номерами.

    Каждая цель ждёт пятно рядом с тем местом, куда она должна была сдвинуться при прежней
    скорости. Пары «цель - пятно» разбираются жадно, начиная с самых близких. Пятно, не
    доставшееся ни одной цели, начинает новую; цель, не получившая пятна `max_missed` кадров
    подряд, забывается."""

    def __init__(self, max_distance: float = 60.0, max_missed: int = 10, smoothing: float = 0.5):
        """Создаёт пустой трекер.

        Args:
            max_distance: дальше скольких пикселей от ожидаемого положения пятно не считается
                продолжением цели; по умолчанию 60.
            max_missed: сколько кадров подряд цель может не находиться, прежде чем будет забыта;
                по умолчанию 10.
            smoothing: доля нового смещения в оценке скорости (0..1]; меньше - скорость
                сглаживается сильнее. По умолчанию 0,5.
        """
        self._max_distance = max_distance
        self._max_missed = max_missed
        self._smoothing = smoothing
        self._targets: list[_Target] = []
        self._next_id = 1

    def update(self, centers: list[tuple[float, float]]) -> list[int]:
        """Обрабатывает один кадр.

        Args:
            centers: центры пятен на этом кадре.

        Returns:
            номер цели для каждого пятна, в том же порядке, что centers.
        """
        pairs = []
        for ti, target in enumerate(self._targets):
            px, py = target.predicted()
            for bi, (x, y) in enumerate(centers):
                distance = math.hypot(x - px, y - py)
                if distance <= self._max_distance:
                    pairs.append((distance, ti, bi))
        pairs.sort()

        ids: list[int | None] = [None] * len(centers)
        used_targets: set[int] = set()
        for _, ti, bi in pairs:
            if ti in used_targets or ids[bi] is not None:
                continue
            used_targets.add(ti)
            target = self._targets[ti]
            steps = target.missed + 1
            x, y = centers[bi]
            step = ((x - target.center[0]) / steps, (y - target.center[1]) / steps)
            a = self._smoothing
            if target.velocity == (0.0, 0.0):
                target.velocity = step  # вторая точка цели: скорость ещё не известна
            else:
                target.velocity = (
                    a * step[0] + (1 - a) * target.velocity[0],
                    a * step[1] + (1 - a) * target.velocity[1],
                )
            target.center = (x, y)
            target.missed = 0
            ids[bi] = target.track_id

        for ti, target in enumerate(self._targets):
            if ti not in used_targets:
                target.missed += 1
        self._targets = [t for t in self._targets if t.missed <= self._max_missed]

        for bi, center in enumerate(centers):
            if ids[bi] is None:
                target = _Target(self._next_id, center, (0.0, 0.0))
                self._next_id += 1
                self._targets.append(target)
                ids[bi] = target.track_id
        return [int(i) for i in ids if i is not None]

    def reset(self) -> None:
        """Забывает все цели; нумерация начинается заново."""
        self._targets = []
        self._next_id = 1


def group_boxes(boxes: list[Box], distance: float) -> list[list[int]]:
    """Склеивает рамки, которые ближе distance пикселей друг к другу (по промежутку между
    рамками, а не между центрами), в группы. Склейка транзитивна: если A рядом с B, а B рядом
    с C, все три попадут в одну группу.

    Args:
        boxes: рамки (x1, y1, x2, y2).
        distance: наибольший промежуток между рамками одной группы в пикселях.

    Returns:
        группы номеров рамок; каждая рамка входит ровно в одну группу.
    """
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            dx = max(b[0] - a[2], a[0] - b[2], 0.0)
            dy = max(b[1] - a[3], a[1] - b[3], 0.0)
            if math.hypot(dx, dy) <= distance:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def merge_group(boxes: list[Box], confs: list[float]) -> tuple[Box, float]:
    """Объединяет рамки одной группы в общую рамку.

    Args:
        boxes: рамки группы.
        confs: уверенность каждой рамки.

    Returns:
        (общая рамка, наибольшая уверенность).
    """
    return (
        (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ),
        max(confs),
    )


class HeadlightsTracker:
    """Ночной детектор с трекером; тот же интерфейс, что у YoloTracker: load(), track(), reset()."""

    def __init__(self, cfg: HeadlightsConfig, zone: Zone | None = None):
        """Запоминает настройки. Фон и маска появятся на первом кадре.

        Args:
            cfg: настройки ночного детектора (раздел [headlights]).
            zone: зона интереса и области, которые нужно игнорировать; None - весь кадр.
        """
        self._cfg = cfg
        self._zone = zone
        self._tracker = CentroidTracker(cfg.max_match_distance, cfg.max_missed_frames)
        self._background: Any = None
        self._mask: Any = None
        self._size: tuple[int, int] | None = None
        self._scale = 1.0  # во сколько раз исходный кадр больше рабочего

    def load(self) -> None:
        """Нечего загружать: метод нужен для совместимости с YoloTracker."""

    def reset(self) -> None:
        """Забывает фон и цели. Нужен после разрыва потока или смены режима день/ночь: старый
        фон к новой картинке не подходит и дал бы ложные пятна."""
        self._background = None
        self._tracker.reset()

    def track(self, frame_bgr: Any) -> list[Observation]:
        """Находит фары на кадре и сопровождает их.

        Args:
            frame_bgr: кадр (массив numpy BGR).

        Returns:
            список Observation с классом 'headlights'; рамки в пикселях исходного кадра.
        """
        blobs = self.detect(frame_bgr)
        ids = self._tracker.update([blob.center for blob in blobs])
        k = self._scale
        return [
            Observation(
                track_id, CLASS_ID, CLASS_NAME, blob.conf, tuple(v * k for v in blob.box)
            )
            for track_id, blob in zip(ids, blobs, strict=True)
        ]

    def detect(self, frame_bgr: Any) -> list[_Blob]:
        """Находит на кадре яркие пятна, которых нет в фоне, и склеивает соседние в цели. Фон
        обновляется после поиска, поэтому на самом первом кадре целей нет.

        Args:
            frame_bgr: кадр (массив numpy BGR).

        Returns:
            найденные цели: рамка (в пикселях рабочего кадра, см. work_width) и уверенность
            (яркость самого яркого пикселя / 255).
        """
        import cv2

        cfg = self._cfg
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > cfg.work_width:
            self._scale = gray.shape[1] / cfg.work_width
            size = (cfg.work_width, round(gray.shape[0] / self._scale))
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        else:
            self._scale = 1.0
        height, width = gray.shape
        if self._size != (width, height):
            self._size = (width, height)
            self._background = None
            self._mask = self._zone.mask(width, height) if self._zone is not None else None
        if self._background is None:
            self._background = gray.astype(np.float32)
            return []

        diff = cv2.subtract(gray, cv2.convertScaleAbs(self._background))
        hot = (gray >= cfg.min_brightness) & (diff >= cfg.min_diff)
        if self._mask is not None:
            hot &= self._mask
        cv2.accumulateWeighted(gray.astype(np.float32), self._background, cfg.background_alpha)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), 8)
        boxes: list[Box] = []
        confs: list[float] = []
        for i in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < cfg.min_area:
                continue
            boxes.append((float(x), float(y), float(x + w), float(y + h)))
            confs.append(float(gray[labels == i].max()) / 255.0 if area < 400 else 1.0)

        blobs = []
        for group in group_boxes(boxes, cfg.pair_distance):
            box, conf = merge_group([boxes[i] for i in group], [confs[i] for i in group])
            w, h = box[2] - box[0], box[3] - box[1]
            if w * h > cfg.max_area:
                continue  # засветка, вспышка ИК-подсветки и т. п., а не фары
            blobs.append(_Blob(box, conf))
        return blobs
