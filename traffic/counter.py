"""Геометрия подсчёта: линии, пересечение которых считается проездом, зона интереса (ROI)
и области кадра, которые нужно игнорировать.

Линий может быть несколько, например на Т-образном перекрёстке: по линии на основной дороге
с каждой стороны от примыкания и линия поперёк примыкающей дороги. Тогда сквозная машина
пересекает обе линии основной дороги, а поворачивающая - одну из них и линию примыкания.

Координаты в конфигурации задаются в долях кадра (0..1): так настройки не зависят от
разрешения потока. В пиксели они переводятся по размеру кадра (Zone.bind).

Модуль не зависит от OpenCV, кроме Zone.mask (маска для ночного детектора).
"""

from collections.abc import Sequence
from typing import Any

Point = tuple[float, float]


def to_pixels(points: Sequence[Sequence[float]], width: int, height: int) -> list[Point]:
    """Переводит точки из долей кадра в пиксели.

    Args:
        points: точки [[x, y], ...], где x и y - доли ширины и высоты кадра (0..1).
        width: ширина кадра в пикселях.
        height: высота кадра в пикселях.

    Returns:
        список точек (x, y) в пикселях.
    """
    return [(float(x) * width, float(y) * height) for x, y in points]


def cross(o: Point, a: Point, b: Point) -> float:
    """Векторное произведение (a - o) x (b - o). Знак показывает, с какой стороны от прямой o-a
    лежит точка b: положительный - слева (в координатах экрана, где y растёт вниз, - справа),
    отрицательный - с другой стороны, ноль - на прямой.

    Args:
        o: начало прямой.
        a: вторая точка прямой.
        b: проверяемая точка.

    Returns:
        значение векторного произведения.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


class CountingLine:
    """Отрезок, пересечение которого траекторией цели считается проездом.

    Направление проезда определяется по тому, с какой стороны отрезка цель была до пересечения.
    directions[0] - переход справа налево, если встать в p1 лицом к p2, directions[1] - обратно.
    На экране это значит: для вертикальной линии, заданной сверху вниз, directions[0] - движение
    слева направо; для горизонтальной линии, заданной слева направо, - движение снизу вверх."""

    def __init__(
        self,
        p1: Point,
        p2: Point,
        directions: tuple[str, str] = ("forward", "backward"),
        name: str = "",
    ):
        """Создаёт линию подсчёта.

        Args:
            p1: начало отрезка в пикселях.
            p2: конец отрезка в пикселях; не должен совпадать с p1.
            directions: названия двух направлений проезда, см. описание класса.
            name: имя линии (для базы, журнала и окна).

        Raises:
            ValueError: p1 и p2 совпадают.
        """
        if p1 == p2:
            raise ValueError("начало и конец линии подсчёта совпадают")
        self.p1 = p1
        self.p2 = p2
        self.directions = directions
        self.name = name

    def crossing(self, a: Point, b: Point) -> str | None:
        """Проверяет, пересёк ли отрезок пути a -> b (центр цели на прошлом и текущем кадре) линию
        подсчёта. Касание линии без перехода на другую сторону пересечением не считается: точка,
        лежащая ровно на линии, относится к стороне, с которой пришла цель, поэтому дрожание
        рамки на линии не даёт ложных проездов.

        Args:
            a: прошлое положение центра цели.
            b: текущее положение центра цели.

        Returns:
            название направления проезда или None, если пересечения не было.
        """
        side_a = cross(self.p1, self.p2, a)
        side_b = cross(self.p1, self.p2, b)
        if side_a == 0 or side_b == 0 or (side_a > 0) == (side_b > 0):
            return None  # обе точки с одной стороны прямой (или на ней)
        # точки по разные стороны прямой; пересекает ли путь именно отрезок, а не его продолжение
        side_p1 = cross(a, b, self.p1)
        side_p2 = cross(a, b, self.p2)
        if (side_p1 > 0 and side_p2 > 0) or (side_p1 < 0 and side_p2 < 0):
            return None
        return self.directions[0] if side_a > 0 else self.directions[1]

    def crossing_from(self, start: Point, previous: Point, current: Point) -> str | None:
        """Проверяет проезд цели, которая появилась в start и за последний кадр сдвинулась
        из previous в current. Проезд засчитывается, когда одновременно:
          - start и current лежат по разные стороны прямой линии: направление берётся от места
            появления, поэтому рамку, дёрнувшуюся назад через линию, это не переворачивает;
          - последний шаг пути previous -> current касается самого отрезка линии, а не его
            продолжения: цель действительно проехала через линию, а не обогнула её (поворот
            с примыкающей дороги мимо короткой линии пересечением не считается).
        Точка ровно на линии для шага считается касанием, поэтому цель, остановившаяся точно на
        линии, будет засчитана следующим шагом.

        Args:
            start: центр цели в момент появления.
            previous: центр цели на прошлом кадре.
            current: центр цели на текущем кадре.

        Returns:
            название направления проезда или None, если проезда не было.
        """
        side_start = cross(self.p1, self.p2, start)
        side_current = cross(self.p1, self.p2, current)
        if side_start == 0 or side_current == 0 or (side_start > 0) == (side_current > 0):
            return None
        side_prev = cross(self.p1, self.p2, previous)
        if side_prev != 0 and (side_prev > 0) == (side_current > 0):
            return None  # последний шаг линию не пересёк
        side_p1 = cross(previous, current, self.p1)
        side_p2 = cross(previous, current, self.p2)
        if (side_p1 > 0 and side_p2 > 0) or (side_p1 < 0 and side_p2 < 0):
            return None  # шаг прошёл мимо отрезка, через его продолжение
        return self.directions[0] if side_start > 0 else self.directions[1]


def inside_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Проверяет, лежит ли точка внутри многоугольника (метод трассировки луча). Пустой
    многоугольник означает «весь кадр», и тогда ответ всегда True.

    Args:
        point: точка (x, y).
        polygon: вершины многоугольника по порядку (не меньше трёх) или пустой список.

    Returns:
        True, если точка внутри (или многоугольник пуст).
    """
    if not polygon:
        return True
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


LineSpec = tuple[str, Sequence[Sequence[float]], Sequence[str]]


class Zone:
    """Всё, что задаёт место подсчёта на кадре: линии подсчёта, зона интереса и игнорируемые
    прямоугольники (например, часы камеры). Хранит координаты в долях кадра и пересчитывает их
    в пиксели при смене размера кадра."""

    def __init__(
        self,
        lines: Sequence[LineSpec] = (),
        roi: Sequence[Sequence[float]] = (),
        ignore: Sequence[Sequence[float]] = (),
    ):
        """Создаёт зону. До первого bind() линии и зона интереса не определены.

        Args:
            lines: линии подсчёта: (имя, [[x1, y1], [x2, y2]] в долях кадра, названия двух
                направлений - см. CountingLine). Пустой список - проезды не считаются.
            roi: вершины зоны интереса в долях кадра; пустой список - весь кадр.
            ignore: прямоугольники [x1, y1, x2, y2] в долях кадра, которые не учитываются.
        """
        self._line_specs = [
            (name, [tuple(p) for p in points], (directions[0], directions[1]))
            for name, points, directions in lines
        ]
        self._roi_fractions = [tuple(p) for p in roi]
        self._ignore_fractions = [tuple(r) for r in ignore]
        self._size: tuple[int, int] | None = None
        self.lines: list[CountingLine] = []
        self.roi: list[Point] = []
        self.ignore: list[tuple[float, float, float, float]] = []

    def bind(self, width: int, height: int) -> None:
        """Пересчитывает координаты в пиксели для кадра заданного размера. Если размер не
        изменился, ничего не делает, поэтому вызывать можно на каждом кадре.

        Args:
            width: ширина кадра в пикселях.
            height: высота кадра в пикселях.
        """
        if self._size == (width, height):
            return
        self._size = (width, height)
        self.lines = []
        for name, points, directions in self._line_specs:
            p1, p2 = to_pixels(points, width, height)
            self.lines.append(CountingLine(p1, p2, directions, name))
        self.roi = to_pixels(self._roi_fractions, width, height)
        self.ignore = [
            (x1 * width, y1 * height, x2 * width, y2 * height)
            for x1, y1, x2, y2 in self._ignore_fractions
        ]

    def contains(self, point: Point) -> bool:
        """Лежит ли точка в зоне интереса и вне игнорируемых прямоугольников. Вызывать после
        bind().

        Args:
            point: точка (x, y) в пикселях.

        Returns:
            True, если точку нужно учитывать.
        """
        x, y = point
        for x1, y1, x2, y2 in self.ignore:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return False
        return inside_polygon(point, self.roi)

    def mask(self, width: int, height: int) -> Any:
        """Маска кадра для ночного детектора: True там, где пиксели нужно учитывать.

        Args:
            width: ширина кадра в пикселях.
            height: высота кадра в пикселях.

        Returns:
            массив numpy bool формы (height, width).
        """
        import cv2
        import numpy as np

        self.bind(width, height)
        if self.roi:
            mask = np.zeros((height, width), np.uint8)
            polygon = np.array([[round(x), round(y)] for x, y in self.roi], np.int32)
            cv2.fillPoly(mask, [polygon], 1)
        else:
            mask = np.ones((height, width), np.uint8)
        for x1, y1, x2, y2 in self.ignore:
            mask[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))] = 0
        return mask.astype(bool)
