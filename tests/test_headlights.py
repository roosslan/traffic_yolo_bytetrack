import numpy as np
import pytest

from traffic.config import HeadlightsConfig
from traffic.counter import Zone
from traffic.headlights import CLASS_NAME, CentroidTracker, HeadlightsTracker, group_boxes


def test_group_boxes_joins_nearby_boxes_transitively():
    boxes = [(0, 0, 2, 2), (5, 0, 7, 2), (10, 0, 12, 2), (100, 100, 102, 102)]
    groups = sorted(sorted(g) for g in group_boxes(boxes, 4))
    assert groups == [[0, 1, 2], [3]]


def test_group_boxes_keeps_far_boxes_apart():
    assert len(group_boxes([(0, 0, 1, 1), (50, 0, 51, 1)], 10)) == 2


def test_centroid_tracker_keeps_the_id_of_a_moving_target():
    tracker = CentroidTracker(max_distance=20)
    ids = [tracker.update([(10.0 * step, 50.0)])[0] for step in range(6)]
    assert len(set(ids)) == 1


def test_centroid_tracker_follows_two_targets_moving_towards_each_other():
    tracker = CentroidTracker(max_distance=30)
    first = tracker.update([(0.0, 0.0), (100.0, 0.0)])
    for step in range(1, 4):
        ids = tracker.update([(10.0 * step, 0.0), (100.0 - 10.0 * step, 0.0)])
        assert ids == first


def test_centroid_tracker_uses_the_velocity_to_bridge_a_missed_frame():
    tracker = CentroidTracker(max_distance=25, max_missed=3)
    [first] = tracker.update([(0.0, 0.0)])
    tracker.update([(20.0, 0.0)])
    tracker.update([])  # цель не нашлась на одном кадре
    # за два кадра цель ушла на 40 px: без учёта скорости это дальше max_distance
    assert tracker.update([(60.0, 0.0)]) == [first]


def test_centroid_tracker_forgets_targets_missing_too_long():
    tracker = CentroidTracker(max_distance=15, max_missed=1)
    [first] = tracker.update([(0.0, 0.0)])
    tracker.update([])
    tracker.update([])
    assert tracker.update([(0.0, 0.0)]) != [first]


def test_centroid_tracker_reset_restarts_numbering():
    tracker = CentroidTracker()
    tracker.update([(0.0, 0.0), (500.0, 0.0)])
    tracker.reset()
    assert tracker.update([(0.0, 0.0)]) == [1]


def dark_frame():
    return np.full((120, 160, 3), 20, np.uint8)


def with_lights(x, y, gap=6):
    frame = dark_frame()
    frame[y : y + 2, x : x + 2] = 255
    frame[y : y + 2, x + gap : x + gap + 2] = 255
    return frame


def test_detects_a_pair_of_headlights_as_one_moving_target():
    pytest.importorskip("cv2")
    tracker = HeadlightsTracker(HeadlightsConfig(pair_distance=10))
    assert tracker.track(dark_frame()) == []  # первый кадр только запоминает фон
    seen = [tracker.track(with_lights(20 + 5 * step, 60)) for step in range(4)]
    assert all(len(obs) == 1 for obs in seen)
    assert len({obs[0].track_id for obs in seen}) == 1
    assert seen[0][0].class_name == CLASS_NAME
    x1, _, x2, _ = seen[0][0].box
    assert (x1, x2) == (20.0, 28.0)  # рамка охватывает обе фары


def test_static_light_becomes_background():
    pytest.importorskip("cv2")
    tracker = HeadlightsTracker(HeadlightsConfig(background_alpha=0.5))
    lamp = with_lights(50, 50)
    tracker.track(dark_frame())
    results = [tracker.track(lamp) for _ in range(10)]
    assert results[0] and not results[-1]


def test_ignored_area_is_masked():
    pytest.importorskip("cv2")
    zone = Zone(ignore=[[0, 0, 0.5, 1]])
    tracker = HeadlightsTracker(HeadlightsConfig(), zone)
    tracker.track(dark_frame())
    assert tracker.track(with_lights(20, 60)) == []  # левая половина кадра игнорируется
    assert len(tracker.track(with_lights(120, 60))) == 1


def test_huge_bright_area_is_not_headlights():
    pytest.importorskip("cv2")
    tracker = HeadlightsTracker(HeadlightsConfig(max_area=100))
    tracker.track(dark_frame())
    flash = dark_frame()
    flash[10:60, 10:60] = 255
    assert tracker.track(flash) == []


def test_large_frames_are_processed_at_work_width_and_boxes_come_back_full_size():
    pytest.importorskip("cv2")
    tracker = HeadlightsTracker(HeadlightsConfig(work_width=160, pair_distance=10))
    big = np.full((480, 640, 3), 20, np.uint8)  # в 4 раза больше рабочего кадра
    tracker.track(big)
    lit = big.copy()
    lit[240:248, 80:88] = 255
    lit[240:248, 104:112] = 255
    [obs] = tracker.track(lit)
    x1, y1, x2, y2 = obs.box
    assert (x1, y1, x2, y2) == (80.0, 240.0, 112.0, 248.0)
