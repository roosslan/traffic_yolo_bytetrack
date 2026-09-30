import numpy as np

from traffic.config import DayNightConfig
from traffic.daynight import DAY, NIGHT, DayNightTracker, ModeSelector, classify, frame_stats
from traffic.detector import Observation

CFG = DayNightConfig(check_interval_sec=5.0, switch_after_sec=30.0)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def gray_frame(level=100):
    return np.full((40, 40, 3), level, np.uint8)


def color_frame():
    frame = gray_frame()
    frame[..., 2] = 200  # красноватый кадр
    return frame


def test_frame_stats():
    assert frame_stats(gray_frame(100)) == (0.0, 100.0)
    colorfulness, _ = frame_stats(color_frame())
    assert colorfulness == 100.0


def test_classify():
    assert classify(0.0, 120.0, CFG) == NIGHT  # серый кадр: ИК-режим
    assert classify(30.0, 10.0, CFG) == NIGHT  # цветной, но тёмный
    assert classify(30.0, 120.0, CFG) == DAY


def test_selector_picks_the_first_mode_immediately():
    selector = ModeSelector(CFG, Clock())
    assert selector.due()
    assert selector.observe(NIGHT)
    assert selector.mode == NIGHT


def test_selector_waits_before_switching():
    clock = Clock()
    selector = ModeSelector(CFG, clock)
    selector.observe(NIGHT)
    clock.now = 5.0
    assert not selector.observe(DAY)  # новый режим только появился
    clock.now = 20.0
    assert not selector.observe(DAY)
    clock.now = 35.0
    assert selector.observe(DAY)  # держится 30 с
    assert selector.mode == DAY


def test_selector_forgets_a_short_flicker():
    clock = Clock()
    selector = ModeSelector(CFG, clock)
    selector.observe(NIGHT)
    clock.now = 5.0
    selector.observe(DAY)  # фары осветили кадр
    clock.now = 10.0
    selector.observe(NIGHT)
    clock.now = 40.0
    assert not selector.observe(DAY)  # отсчёт начался заново с 40 с
    assert selector.mode == NIGHT


def test_selector_checks_only_every_interval():
    clock = Clock()
    selector = ModeSelector(CFG, clock)
    selector.observe(DAY)
    clock.now = 4.0
    assert not selector.due()
    clock.now = 5.0
    assert selector.due()


def test_fixed_mode_never_changes():
    selector = ModeSelector(DayNightConfig(mode="night"), Clock())
    assert selector.mode == NIGHT
    assert not selector.due()


class FakeDetector:
    def __init__(self, name):
        self.name, self.loaded, self.resets, self.frames = name, 0, 0, 0

    def load(self):
        self.loaded += 1

    def track(self, frame):
        self.frames += 1
        return [Observation(1, 0, self.name, 0.9, (0, 0, 1, 1))]

    def reset(self):
        self.resets += 1


def test_tracker_uses_the_detector_of_the_current_mode_and_reports_switches():
    clock = Clock()
    day, night, switches = FakeDetector("day"), FakeDetector("night"), []
    tracker = DayNightTracker(CFG, day, night, on_switch=switches.append, clock=clock)
    tracker.load()
    assert (day.loaded, night.loaded) == (1, 1)  # в режиме auto загружаются оба

    assert tracker.track(gray_frame())[0].class_name == "night"
    assert switches == [NIGHT]
    for t in (5.0, 40.0):
        clock.now = t
        tracker.track(color_frame())
    assert tracker.mode == DAY and switches == [NIGHT, DAY]
    assert tracker.track(color_frame())[0].class_name == "day"
    assert night.resets >= 1  # при смене режима детекторы забывают цели


def test_fixed_mode_loads_only_its_detector():
    day, night = FakeDetector("day"), FakeDetector("night")
    tracker = DayNightTracker(DayNightConfig(mode="day"), day, night)
    tracker.load()
    assert (day.loaded, night.loaded) == (1, 0)
    assert tracker.track(gray_frame())[0].class_name == "day"  # серый кадр не важен
