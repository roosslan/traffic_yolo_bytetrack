import pytest

from traffic.counter import CountingLine, Zone, inside_polygon, to_pixels

# Вертикальная линия x = 50 сверху вниз: directions[0] - движение слева направо.
LINE = CountingLine((50.0, 0.0), (50.0, 100.0), ("right", "left"))


def test_crossing_left_to_right_gives_the_first_direction():
    assert LINE.crossing((40.0, 50.0), (60.0, 50.0)) == "right"


def test_crossing_right_to_left_gives_the_second_direction():
    assert LINE.crossing((60.0, 50.0), (40.0, 50.0)) == "left"


def test_horizontal_line_left_to_right_counts_upward_motion_first():
    line = CountingLine((0.0, 50.0), (100.0, 50.0), ("up", "down"))
    assert line.crossing((50.0, 60.0), (50.0, 40.0)) == "up"
    assert line.crossing((50.0, 40.0), (50.0, 60.0)) == "down"


@pytest.mark.parametrize(
    "a, b",
    [
        ((40.0, 50.0), (45.0, 50.0)),  # не дошла до линии
        ((40.0, 50.0), (50.0, 50.0)),  # встала ровно на линию: это ещё не переход
        ((50.0, 50.0), (60.0, 50.0)),  # ушла с линии в ту сторону, где и так была
        ((40.0, 150.0), (60.0, 150.0)),  # пересекла продолжение линии, а не сам отрезок
    ],
)
def test_no_crossing(a, b):
    assert LINE.crossing(a, b) is None


def test_crossing_through_the_end_of_the_segment_counts():
    assert LINE.crossing((40.0, 100.0), (60.0, 100.0)) == "right"


def test_crossing_from_takes_the_direction_from_the_start():
    # рамка дёрнулась назад через линию: от места появления это всё ещё не проезд назад
    assert LINE.crossing_from((20.0, 50.0), (53.0, 50.0), (47.0, 50.0)) is None
    assert LINE.crossing_from((20.0, 50.0), (47.0, 50.0), (58.0, 50.0)) == "right"


def test_crossing_from_needs_the_last_step_to_pass_the_segment():
    # место появления и текущая точка по разные стороны, но шаг прошёл ниже конца отрезка
    assert LINE.crossing_from((40.0, 50.0), (45.0, 150.0), (60.0, 150.0)) is None
    # шаг уже целиком по ту сторону линии: пересечение было раньше
    assert LINE.crossing_from((40.0, 50.0), (60.0, 50.0), (70.0, 50.0)) is None


def test_crossing_from_counts_a_turn_that_the_straight_chord_would_miss():
    short = CountingLine((30.0, 60.0), (30.0, 80.0), ("east", "west"))
    # машина спустилась сверху (с примыкания) и повернула налево по основной дороге
    assert short.crossing((50.0, 10.0), (25.0, 70.0)) is None  # хорда мимо короткой линии
    assert short.crossing_from((50.0, 10.0), (40.0, 70.0), (25.0, 70.0)) == "west"


def test_crossing_from_counts_a_target_that_stopped_exactly_on_the_line():
    assert LINE.crossing_from((40.0, 50.0), (50.0, 50.0), (55.0, 50.0)) == "right"


def test_degenerate_line_is_rejected():
    with pytest.raises(ValueError):
        CountingLine((1.0, 1.0), (1.0, 1.0))


def test_to_pixels():
    assert to_pixels([[0.5, 0.25]], 200, 100) == [(100.0, 25.0)]


SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_inside_polygon():
    assert inside_polygon((5.0, 5.0), SQUARE)
    assert not inside_polygon((15.0, 5.0), SQUARE)
    assert inside_polygon((1e6, 1e6), [])  # пустой многоугольник - весь кадр


def test_zone_scales_to_the_frame_and_rescales_on_size_change():
    zone = Zone(
        [("mid", [[0.5, 0.0], [0.5, 1.0]], ["a", "b"]), ("top", [[0, 0], [1, 0]], ["c", "d"])],
        roi=[[0, 0], [1, 0], [1, 0.5]],
    )
    zone.bind(200, 100)
    [mid, top] = zone.lines
    assert (mid.name, mid.p1, mid.p2) == ("mid", (100.0, 0.0), (100.0, 100.0))
    assert (top.name, top.directions) == ("top", ("c", "d"))
    zone.bind(400, 100)
    assert zone.lines[0].p1 == (200.0, 0.0)


def test_zone_without_a_line():
    zone = Zone()
    zone.bind(10, 10)
    assert zone.lines == []
    assert zone.contains((5.0, 5.0))


def test_zone_contains_respects_roi_and_ignored_rectangles():
    zone = Zone(roi=[[0, 0], [1, 0], [1, 1], [0, 1]], ignore=[[0, 0, 0.5, 0.1]])
    zone.bind(100, 100)
    assert zone.contains((50.0, 50.0))
    assert not zone.contains((10.0, 5.0))  # в игнорируемом прямоугольнике


def test_zone_mask():
    pytest.importorskip("cv2")
    zone = Zone(roi=[[0, 0], [0.5, 0], [0.5, 1], [0, 1]], ignore=[[0, 0, 0.25, 0.25]])
    mask = zone.mask(8, 8)
    assert mask.shape == (8, 8)
    assert not mask[0, 0]  # игнорируется
    assert mask[5, 1]  # в зоне
    assert not mask[5, 7]  # вне зоны
