"""Выбор режима день/ночь и детектор, который переключается между YOLO и фарами.

Днём машины хорошо видны, и их находит нейросеть (YOLO + ByteTrack). Ночью камера за
стеклом видит только фары, поэтому работает детектор фар. В режиме auto режим выбирается
по самой картинке:

* **ночь**: кадр почти серый (камера перешла в ИК-режим и отдаёт чёрно-белую картинку) или
  тёмный;
* **день**: всё остальное.

Чтобы режим не прыгал туда-сюда в сумерках или от фар проезжающей машины, новый режим
включается, только если он держится `switch_after_sec` секунд подряд.
"""

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

from traffic.config import DayNightConfig
from traffic.detector import Observation

log = logging.getLogger(__name__)

DAY = "day"
NIGHT = "night"


class Tracker(Protocol):
    def load(self) -> None:
        """Загрузить модель (если она нужна)."""

    def track(self, frame_bgr: Any) -> list[Observation]:
        """Найти и сопроводить цели на одном кадре.

        Args:
            frame_bgr: кадр (массив numpy BGR).

        Returns:
            список Observation.
        """

    def reset(self) -> None:
        """Забыть все цели."""


def frame_stats(frame_bgr: Any, step: int = 4) -> tuple[float, float]:
    """Оценивает цветность и яркость кадра по каждому step-му пикселю (для скорости).

    Args:
        frame_bgr: кадр (массив numpy BGR).
        step: шаг прореживания по обеим осям; по умолчанию 4.

    Returns:
        (цветность, яркость): средняя по пикселям наибольшая разница между каналами (0 для
        серого кадра) и средняя яркость (0..255).
    """
    import numpy as np

    small = frame_bgr[::step, ::step].astype(np.int16)
    colorfulness = float((small.max(axis=2) - small.min(axis=2)).mean())
    brightness = float(small.mean())
    return colorfulness, brightness


def classify(colorfulness: float, brightness: float, cfg: DayNightConfig) -> str:
    """Решает по одному кадру, день на нём или ночь.

    Args:
        colorfulness: цветность кадра (см. frame_stats).
        brightness: средняя яркость кадра (0..255).
        cfg: пороги (раздел [daynight]).

    Returns:
        DAY или NIGHT.
    """
    if colorfulness < cfg.color_threshold or brightness < cfg.dark_threshold:
        return NIGHT
    return DAY


class ModeSelector:
    """Выбирает режим с задержкой: новый режим включается, только если кадры подряд
    признавались им не меньше switch_after_sec секунд."""

    def __init__(self, cfg: DayNightConfig, clock: Callable[[], float] = time.monotonic):
        """Создаёт выбор режима. Если режим задан явно (day или night), он никогда не меняется.

        Args:
            cfg: настройки раздела [daynight].
            clock: монотонные часы; по умолчанию time.monotonic, в тестах подменяются.
        """
        self._cfg = cfg
        self._clock = clock
        self.mode: str | None = None if cfg.mode == "auto" else cfg.mode
        self._candidate: str | None = None
        self._candidate_since = 0.0
        self._checked_at: float | None = None

    def due(self) -> bool:
        """Пора ли оценить очередной кадр: в режиме auto раз в check_interval_sec секунд, а пока
        режим ещё не выбран - на каждом кадре.

        Returns:
            True, если нужно вызвать observe().
        """
        if self._cfg.mode != "auto":
            return False
        if self.mode is None or self._checked_at is None:
            return True
        return self._clock() - self._checked_at >= self._cfg.check_interval_sec

    def observe(self, verdict: str) -> bool:
        """Учитывает решение по очередному кадру.

        Args:
            verdict: DAY или NIGHT для этого кадра (см. classify).

        Returns:
            True, если режим сменился (в том числе самый первый выбор режима).
        """
        now = self._clock()
        self._checked_at = now
        if self.mode is None:  # первый кадр: ждать нечего
            self.mode = verdict
            return True
        if verdict == self.mode:
            self._candidate = None
            return False
        if verdict != self._candidate:
            self._candidate, self._candidate_since = verdict, now
        if now - self._candidate_since >= self._cfg.switch_after_sec:
            self.mode, self._candidate = verdict, None
            return True
        return False


class DayNightTracker:
    """Детектор с тем же интерфейсом, что YoloTracker, который сам переключается между дневным
    (YOLO) и ночным (фары) детектором."""

    def __init__(
        self,
        cfg: DayNightConfig,
        day: Tracker,
        night: Tracker,
        on_switch: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Создаёт переключаемый детектор.

        Args:
            cfg: настройки раздела [daynight].
            day: дневной детектор (YoloTracker).
            night: ночной детектор (HeadlightsTracker).
            on_switch: вызывается при каждой смене режима с новым режимом (DAY или NIGHT) до того,
                как новый детектор получит кадр. Нужен, чтобы закрыть цели старого режима: номера
                у двух детекторов свои и могут совпадать.
            clock: монотонные часы; по умолчанию time.monotonic.
        """
        self._cfg = cfg
        self._selector = ModeSelector(cfg, clock)
        self._detectors = {DAY: day, NIGHT: night}
        self._on_switch = on_switch
        self._loaded: set[str] = set()

    @property
    def mode(self) -> str | None:
        """Текущий режим: DAY, NIGHT или None, пока не пришёл ни один кадр в режиме auto.

        Returns:
            режим.
        """
        return self._selector.mode

    def load(self) -> None:
        """Загружает детекторы. В режиме auto загружаются оба (чтобы смена режима не ждала
        загрузки модели), в явном режиме - только нужный.
        """
        wanted = [self._selector.mode] if self._selector.mode else [DAY, NIGHT]
        for mode in wanted:
            self._detectors[mode].load()
            self._loaded.add(mode)

    def track(self, frame_bgr: Any) -> list[Observation]:
        """Выбирает режим (если пора) и прогоняет кадр через детектор этого режима.

        Args:
            frame_bgr: кадр (массив numpy BGR).

        Returns:
            список Observation от детектора текущего режима.
        """
        if self._selector.due():
            verdict = classify(*frame_stats(frame_bgr), self._cfg)
            if self._selector.observe(verdict):
                mode = self._selector.mode
                log.info("Режим: %s", "день (YOLO)" if mode == DAY else "ночь (фары)")
                for detector in self._detectors.values():
                    detector.reset()
                if self._on_switch is not None:
                    self._on_switch(mode)
        mode = self._selector.mode
        assert mode is not None
        if mode not in self._loaded:
            self._detectors[mode].load()
            self._loaded.add(mode)
        return self._detectors[mode].track(frame_bgr)

    def reset(self) -> None:
        """Сбрасывает оба детектора (после разрыва потока)."""
        for detector in self._detectors.values():
            detector.reset()
