"""Futsal analytics core: pitch geometry, camera calibration, two-camera fusion and player metrics.

Coordinate convention used everywhere (metres):
    x runs along the touchline, 0 .. length; y runs along the goal line, 0 .. width.
    Goals are centred on x = 0 and x = length. The bench / substitution zones are on y = 0.
"""

from .court import Court

__all__ = ["Court"]
