"""Что делать с событиями целей: снимок, запись в ClickHouse, строка в журнал, счётчики."""

import logging
import threading
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from typing import Any

from traffic.detector import Observation
from traffic.snapshots import SnapshotWriter
from traffic.storage import (
    CROSSINGS,
    EVENTS,
    POINTS,
    TRACKS,
    ClickHouseSink,
    crossing_row,
    event_row,
    point_row,
    track_row,
)
from traffic.tracks import Crossing, Track, TrackView

log = logging.getLogger(__name__)


class TargetRecorder:
    """Обработчики событий `TrackRegistry`.

    И снимки, и база необязательны: без них цели только пишутся в журнал."""

    def __init__(
        self,
        camera: str,
        session_id: uuid.UUID,
        sink: ClickHouseSink | None = None,
        snapshots: SnapshotWriter | None = None,
        annotate: Callable[[Any, list[TrackView]], Any] | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        """Создаёт обработчик событий целей.

        Args:
            camera: имя камеры ([camera] name); записывается во все строки ClickHouse.
            session_id: уникальный номер текущего запуска программы. Позволяет различать цели разных
                запусков: номера от трекера после перезапуска начинаются заново.
            sink: запись в ClickHouse; None - события пишутся только в журнал.
            snapshots: сохранение снимков на диск; None - снимки не делаются.
            annotate: функция (кадр, цели) -> изображение, рисующая цели на копии кадра для снимка;
                None - снимок сохраняется без рамок.
            now: функция, возвращающая время для имени файла снимка; None (по умолчанию) - берётся
                момент проезда по местному времени. Нужна в основном тестам.
        """
        self._camera = camera
        self._session_id = session_id
        self._sink = sink
        self._snapshots = snapshots
        self._annotate = annotate
        self._now = now
        self._counts: Counter[tuple[str, str]] = Counter()
        self._counts_lock = threading.Lock()  # счётчики читает окно из другого потока

    def counts(self) -> dict[tuple[str, str], int]:
        """Сколько проездов засчитано с начала работы, по линиям и направлениям.

        Returns:
            {(линия, направление): число проездов}.
        """
        with self._counts_lock:
            return dict(self._counts)

    def on_captured(self, track: Track, frame: Any, views: list[TrackView]) -> None:
        """Вызывается реестром, когда цель подтверждена (захвачена). Пишет строку в журнал
        (на уровне DEBUG: целей много, главное - проезды) и добавляет событие 'captured'
        в ClickHouse.

        Args:
            track: захваченная цель.
            frame: кадр, на котором цель подтверждена; здесь не используется.
            views: все цели на этом кадре; здесь не используются.
        """
        log.debug(
            "Захват: %s #%d (уверенность %.2f)", track.class_name, track.track_id, track.conf_max
        )
        if self._sink is not None:
            ts = track.captured_at if track.captured_at is not None else track.first_seen
            self._sink.add(EVENTS, event_row(self._camera, self._session_id, track, "captured", ts))

    def on_crossed(
        self, track: Track, crossing: Crossing, frame: Any, views: list[TrackView]
    ) -> None:
        """Вызывается реестром, когда захваченная цель пересекла линию подсчёта. Сохраняет снимок
        кадра (с нарисованными целями, если задан annotate) и записывает путь к нему
        в crossing.snapshot (и в track.snapshot, если это первый снимок цели), увеличивает
        счётчик линии и направления, пишет строку в журнал и добавляет проезд в ClickHouse.
        Ошибка при сохранении снимка пишется в журнал и не мешает записи самого проезда.

        Args:
            track: цель, пересёкшая линию.
            crossing: пересечение: линия, направление, время; в него записывается путь к снимку.
            frame: кадр, на котором засчитан проезд (массив numpy BGR); None - снимок не делается.
            views: все цели на этом кадре, чтобы нарисовать их на снимке.
        """
        if self._snapshots is not None and frame is not None:
            image = self._annotate(frame, views) if self._annotate else frame
            try:
                path = self._snapshots.save(
                    image,
                    self._snapshot_time(crossing),
                    f"{track.class_name}-{crossing.line}",
                    track.track_id,
                )
                crossing.snapshot = str(path)
                track.snapshot = track.snapshot or crossing.snapshot
            except Exception:
                # Снимок не должен мешать записи самого события.
                log.exception("Не удалось сохранить снимок")
        key = (crossing.line, crossing.direction)
        with self._counts_lock:
            self._counts[key] += 1
            total = self._counts[key]
        log.info(
            "Проезд: %s #%d, линия %s, направление %s (всего %d)%s",
            track.class_name,
            track.track_id,
            crossing.line,
            crossing.direction,
            total,
            f", снимок {crossing.snapshot}" if crossing.snapshot else "",
        )
        if self._sink is not None:
            self._sink.add(
                CROSSINGS, crossing_row(self._camera, self._session_id, track, crossing)
            )

    def _snapshot_time(self, crossing: Crossing) -> datetime:
        """Выбирает время для имени файла снимка: результат функции now, если она задана, иначе
        момент пересечения линии по местному времени.

        Args:
            crossing: пересечение, для которого делается снимок.

        Returns:
            datetime по местному времени (без указания часового пояса).
        """
        if self._now is not None:
            return self._now()
        return datetime.fromtimestamp(crossing.ts)

    def on_point(self, track: Track, obs: Observation, ts: float) -> None:
        """Вызывается реестром для очередной точки траектории захваченной цели (не чаще раза
        в points_interval_sec). Добавляет строку с рамкой и уверенностью в таблицу
        track_points. Без ClickHouse ничего не делает.

        Args:
            track: цель, к которой относится точка.
            obs: наблюдение цели на текущем кадре: рамка и уверенность.
            ts: время кадра в секундах с 1970 г. (UTC).
        """
        if self._sink is not None:
            self._sink.add(POINTS, point_row(self._camera, track, obs, ts))

    def on_lost(self, track: Track, reason: str) -> None:
        """Вызывается реестром, когда сопровождение цели закончено. Пишет итог в журнал
        (длительность, пройденный путь, причина) и добавляет в ClickHouse событие 'lost'
        в таблицу events и итоговую строку по цели в таблицу tracks.

        Args:
            track: цель со всей накопленной статистикой.
            reason: почему сопровождение закончено: 'lost' (цель пропала из кадра), 'gap' (разрыв
                видеопотока) или 'shutdown' (остановка программы).
        """
        log.debug(
            "Потеря: %s #%d после %.1f с сопровождения, путь %.0f px (%s)",
            track.class_name,
            track.track_id,
            track.duration,
            track.path_length,
            reason,
        )
        if self._sink is not None:
            self._sink.add(
                EVENTS, event_row(self._camera, self._session_id, track, "lost", track.last_seen)
            )
            self._sink.add(TRACKS, track_row(self._camera, self._session_id, track, reason))
