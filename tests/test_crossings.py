"""Проезды через линии подсчёта в реестре целей."""

from traffic.counter import Zone
from traffic.detector import Observation
from traffic.tracks import TrackRegistry


class Frame:
    """Заглушка кадра: реестру нужен только размер."""

    shape = (100, 100, 3)


FRAME = Frame()

# вертикальная линия посередине кадра, сверху вниз: "right" - движение слева направо
MIDDLE = ("mid", [[0.5, 0.0], [0.5, 1.0]], ["right", "left"])


def at(track_id, x, y=50.0, name="car", size=4.0):
    return Observation(track_id, 2, name, 0.9, (x - size, y - size, x + size, y + size))


def make(lines=(MIDDLE,), roi=(), merge_distance=None, confirm_hits=2):
    zone = Zone(lines, roi=roi)
    crossings = []
    clock = [0.0]
    registry = TrackRegistry(
        on_crossed=lambda track, crossing, frame, views: crossings.append(
            (track.track_id, crossing.line, crossing.direction, crossing.ts)
        ),
        zone=zone,
        merge_distance=merge_distance,
        confirm_hits=confirm_hits,
        clock=lambda: clock[0],
        wall_clock=lambda: clock[0],
    )

    def step(t, observations):
        clock[0] = t
        registry.update(observations, FRAME, now=t)

    return registry, crossings, step


def test_crossing_is_reported_once_with_line_direction_and_time():
    _, crossings, step = make()
    for i, x in enumerate([30, 40, 45, 55, 60, 70, 40, 60]):  # потом ещё туда-сюда
        step(i * 0.1, [at(1, x)])
    assert crossings == [(1, "mid", "right", 0.30000000000000004)]


def test_opposite_direction():
    _, crossings, step = make()
    for i, x in enumerate([70, 60, 45, 30]):
        step(i * 0.1, [at(1, x)])
    assert [c[2] for c in crossings] == ["left"]


def test_crossing_before_capture_is_reported_at_capture():
    _, crossings, step = make(confirm_hits=3)
    step(0.0, [at(1, 45)])
    step(0.1, [at(1, 55)])  # пересекла, но ещё не подтверждена
    assert crossings == []
    step(0.2, [at(1, 60)])
    assert [(c[0], c[2]) for c in crossings] == [(1, "right")]
    assert crossings[0][3] == 0.1  # время самого пересечения, а не захвата


def test_unconfirmed_target_is_never_counted():
    _, crossings, step = make(confirm_hits=5)
    step(0.0, [at(1, 45)])
    step(0.1, [at(1, 55)])
    step(5.0, [])  # пропала, не дойдя до захвата
    assert crossings == []


def test_jitter_of_the_box_does_not_flip_the_direction():
    """Центр ночной рамки прыгает назад, когда загораются габариты: считается от старта."""
    _, crossings, step = make()
    for i, x in enumerate([20, 30, 40, 48, 53, 47, 58]):
        step(i * 0.1, [at(1, x)])
    assert [c[2] for c in crossings] == ["right"]


def test_targets_outside_the_roi_are_ignored():
    # зона интереса - только нижняя половина кадра
    _, crossings, step = make(roi=[[0, 0.5], [1, 0.5], [1, 1], [0, 1]])
    for i, x in enumerate([30, 45, 55, 70]):
        step(i * 0.1, [at(1, x, y=20.0), at(2, x, y=80.0)])
    assert [c[0] for c in crossings] == [2]


# Т-образный перекрёсток: основная дорога горизонтально на y = 70, примыкание сверху по x = 50.
T_JUNCTION = (
    ("west", [[0.3, 0.6], [0.3, 0.8]], ["east", "west"]),
    ("east", [[0.7, 0.6], [0.7, 0.8]], ["east", "west"]),
    # горизонтальная линия поперёк примыкания, слева направо: [0] - движение вверх (въезд)
    ("side", [[0.4, 0.4], [0.6, 0.4]], ["in", "out"]),
)


def drive(step, track_id, points):
    for i, (x, y) in enumerate(points):
        step(i * 0.1, [at(track_id, x, y)])


def test_through_traffic_crosses_both_main_road_lines():
    _, crossings, step = make(lines=T_JUNCTION)
    drive(step, 1, [(10 + 10 * i, 70) for i in range(9)])  # слева направо по основной
    assert [(c[1], c[2]) for c in crossings] == [("west", "east"), ("east", "east")]


def test_turn_from_the_side_road():
    _, crossings, step = make(lines=T_JUNCTION)
    # спускается по примыканию и поворачивает налево (на запад)
    drive(step, 1, [(50, 10), (50, 25), (50, 45), (50, 60), (40, 70), (25, 70), (10, 70)])
    assert [(c[1], c[2]) for c in crossings] == [("side", "out"), ("west", "west")]


def test_turn_into_the_side_road():
    _, crossings, step = make(lines=T_JUNCTION)
    # едет с востока и сворачивает в примыкание
    drive(step, 1, [(95, 70), (80, 70), (65, 70), (55, 65), (50, 50), (50, 30), (50, 15)])
    assert [(c[1], c[2]) for c in crossings] == [("east", "west"), ("side", "in")]


def test_rear_lights_following_a_counted_car_are_merged():
    """Фары (цель 1) и габариты (цель 2) одной машины пересекают линию друг за другом."""
    _, crossings, step = make(merge_distance={"headlights": 0.3})  # 30 px при ширине 100
    for i, x in enumerate([40, 45, 55, 60, 65, 70, 75, 80]):
        step(i * 0.1, [at(1, x, name="headlights"), at(2, x - 20, name="headlights")])
    assert [c[0] for c in crossings] == [1]


def test_far_apart_cars_are_both_counted():
    _, crossings, step = make(merge_distance={"headlights": 0.1})
    for i, x in enumerate([40, 45, 55, 60, 65, 70, 75, 80]):
        step(i * 0.1, [at(1, x, name="headlights"), at(2, x - 20, name="headlights")])
    assert [c[0] for c in crossings] == [1, 2]


def test_merge_applies_only_to_the_configured_class():
    _, crossings, step = make(merge_distance={"headlights": 0.3})
    for i, x in enumerate([40, 45, 55, 60, 65, 70, 75, 80]):
        step(i * 0.1, [at(1, x), at(2, x - 20)])  # класс car
    assert [c[0] for c in crossings] == [1, 2]


def test_merge_is_per_line():
    """Задние огни не склеиваются с машиной, засчитанной на другой линии."""
    lines = (
        ("low", [[0.5, 0.25], [0.5, 1.0]], ["right", "left"]),
        ("other", [[0.5, 0.0], [0.5, 0.2]], ["right", "left"]),
    )
    _, crossings, step = make(lines=lines, merge_distance={"headlights": 0.3})
    for i, x in enumerate([40, 45, 55, 60, 65, 70, 75, 80]):
        step(
            i * 0.1,
            [at(1, x, y=10.0, name="headlights"), at(2, x - 20, y=30.0, name="headlights")],
        )
    assert [(c[0], c[1]) for c in crossings] == [(1, "other"), (2, "low")]


def test_no_lines_means_no_crossings():
    registry = TrackRegistry(
        on_crossed=lambda *args: (_ for _ in ()).throw(AssertionError("не должно вызываться")),
        zone=Zone(),
        confirm_hits=1,
    )
    for i, x in enumerate([30, 70]):
        registry.update([at(1, x)], FRAME, now=float(i))
