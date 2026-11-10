"""Small 2D geometry helpers. All distances are in metres, angles in degrees."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def scale(self, k: float) -> "Point":
        return Point(self.x * k, self.y * k)

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Point":
        n = self.length()
        if n < 1e-9:
            return Point(0.0, 0.0)
        return Point(self.x / n, self.y / n)

    def rotated_deg(self, deg: float) -> "Point":
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        return Point(self.x * c - self.y * s, self.x * s + self.y * c)


def angle_of(v: Point) -> float:
    """Direction of vector v, degrees in [0, 360)."""
    return math.degrees(math.atan2(v.y, v.x)) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two headings, in [0, 180]."""
    d = abs((a - b) % 360.0)
    return 360.0 - d if d > 180.0 else d


def polar(deg: float, r: float) -> Point:
    rad = math.radians(deg)
    return Point(r * math.cos(rad), r * math.sin(rad))


def quantize_heading(deg: float, sectors: int = 8) -> int:
    """Which of `sectors` compass sectors a heading falls into."""
    step = 360.0 / sectors
    return int((deg + step / 2) // step) % sectors
