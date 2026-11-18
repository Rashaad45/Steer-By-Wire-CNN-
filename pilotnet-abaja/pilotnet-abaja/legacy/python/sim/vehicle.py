"""Kinematic bicycle model + pure-pursuit expert. Mirrors matlab/bicycle_step.m
and pure_pursuit.m."""
from __future__ import annotations

import math

from . import config as C
from .course import Route, cross_track, nearest_on_route


def bicycle_step(state, delta_cmd: float, v: float, dt: float = C.DT):
    """state = [x, y, psi, delta] (rear-axle reference). Returns new state.

    Steering actuator: first-order lag (tau) with slew-rate limit, then the
    kinematic bicycle  x' = v cos(psi), y' = v sin(psi), psi' = v tan(delta)/L.
    Semi-implicit Euler: the updated steering angle is used for the pose.
    """
    x, y, psi, d = state
    delta_cmd = max(-C.MAX_STEER, min(C.MAX_STEER, delta_cmd))
    rate = (delta_cmd - d) / C.STEER_TAU
    rate = max(-C.STEER_RATE, min(C.STEER_RATE, rate))
    d = max(-C.MAX_STEER, min(C.MAX_STEER, d + rate * dt))
    x += v * math.cos(psi) * dt
    y += v * math.sin(psi) * dt
    psi += v / C.WHEELBASE * math.tan(d) * dt
    psi = math.atan2(math.sin(psi), math.cos(psi))
    return [x, y, psi, d]


def pure_pursuit(route: Route, x: float, y: float, psi: float, v: float, hint: int):
    """Pure pursuit (Coulter, 1992) on the planned route.

    Returns (kappa, nearest_index, cte). kappa = 2*y_L / L_d^2 where y_L is the
    lateral coordinate of the look-ahead point in the vehicle frame.
    """
    i = nearest_on_route(route, x, y, hint)
    ld = C.LOOKAHEAD_MIN + C.LOOKAHEAD_GAIN * v
    s_t = route.s[i] + ld
    j = i
    n = len(route.s)
    while j < n - 1 and route.s[j] < s_t:
        j += 1
    px, py = route.pts[j]
    dx = (px - x) * math.cos(psi) + (py - y) * math.sin(psi)
    dy = -(px - x) * math.sin(psi) + (py - y) * math.cos(psi)
    L2 = max(dx * dx + dy * dy, 1e-6)
    kappa = 2.0 * dy / L2
    kappa = max(-C.KAPPA_MAX, min(C.KAPPA_MAX, kappa))
    return kappa, i, cross_track(route, i, x, y)


def kappa_to_delta(kappa: float) -> float:
    return math.atan(C.WHEELBASE * kappa)
