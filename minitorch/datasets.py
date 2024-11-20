import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates N random points in the unit square.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        List: A list of N tuples, where each tuple contains two float values representing the coordinates of a point in the unit square.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a simple binary classification dataset with N random points in the unit square. The label is 1 if the x-coordinate is less than 0.5, otherwise 0.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels. It includes the number of points `N`, the list of points `X`, and the list of labels `y`.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a diagonal binary classification dataset with N random points in the unit square. The label is 1 if the sum of the x- and y-coordinates is less than 0.5, otherwise 0. The diagonal line serves as the classification boundary.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels. It includes the number of points `N`, the list of points `X`, and the list of labels `y`.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a split binary classification dataset with N random points in the unit square. The label is 1 if the x-coordinate is less than 0.2 or greater than 0.8, otherwise 0. These vertical lines serve as the classification boundary.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels. It includes the number of points `N`, the list of points `X`, and the list of labels `y`.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates an XOR binary classification dataset with N random points in the unit square. The label is 1 if the x-coordinate is less than 0.5 and the y-coordinate is greater than 0.5, or if the x-coordinate is greater than 0.5 and the y-coordinate is less than 0.5, otherwise 0. The diagonal quadrant boundaries serve as classification boundaries.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels. It includes the number of points `N`, the list of points `X`, and the list of labels `y`.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a circle binary classification dataset with N random points in the unit square. The label is 1 if the point is within a circle of radius 0.1 centered at (0.5, 0.5), otherwise 0. The circle serves as the classification boundary.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels. It includes the number of points `N`, the list of points `X`, and the list of labels `y`.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a spiral binary classification dataset with N random points. The function generates `N` points in two wrapped spiral patterns, where the first spiral is made up of half the points labeled 0 and the second one is the other half labeled 1. These spiral patterns serve as the classification boundary.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and labels. It includes the number of points `N`, the list of points `X`, and the list of labels `y2`.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
