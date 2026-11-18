"""Shared simulator configuration.

Every value here is duplicated in matlab/abaja_config.m. Keep the two files in
sync: the renderer, course and texture are designed to produce identical
images in Python and MATLAB, so a dataset generated in one can be used in the
other.
"""
import math

# ---------------- course (BAJA-style off-road loop with one fork) ----------
# node: name -> (x [m], y [m], heading [deg])
NODES = {
    "S":  (60.0, 12.0, 180.0),
    "J1": (20.0, 45.0, 90.0),
    "A":  (55.0, 83.0, 0.0),
    "B":  (58.0, 58.0, 0.0),
    "J2": (108.0, 50.0, -90.0),
}
# directed edges: (from, to, [intermediate control points])
EDGES = [
    ("S",  "J1", [(36.0, 13.0), (22.0, 26.0)]),
    ("J1", "A",  [(12.0, 62.0), (22.0, 80.0)]),          # left branch
    ("A",  "J2", [(88.0, 84.0), (105.0, 72.0)]),
    ("J1", "B",  [(26.0, 55.0), (40.0, 59.0)]),          # right branch
    ("B",  "J2", [(80.0, 57.0), (98.0, 62.0)]),
    ("J2", "S",  [(104.0, 28.0), (88.0, 14.0)]),
]
TRACK_WIDTH = 6.0          # [m]
PATH_DS = 0.25             # centreline resampling step [m]

# ---------------- map raster -------------------------------------------------
MAP_X0, MAP_Y0 = -15.0, -15.0
MAP_W_M, MAP_H_M = 150.0, 120.0
MAP_RES = 0.10             # [m / cell]

# ---------------- vehicle (aBAJA-class ATV, kinematic bicycle) ---------------
WHEELBASE = 1.40           # [m]
MAX_STEER = math.radians(30.0)
STEER_TAU = 0.15           # first-order steering actuator lag [s]
STEER_RATE = math.radians(120.0)   # actuator slew limit [rad/s]
KAPPA_MAX = math.tan(MAX_STEER) / WHEELBASE   # 0.412 1/m -> label normalisation
SPEED = 5.0                # nominal speed [m/s]
DT = 0.05                  # simulation step [s] (20 Hz)

# ---------------- camera -----------------------------------------------------
IMG_W, IMG_H = 200, 66     # PilotNet input
SUPERSAMPLE = 2            # render at 2x then box-filter down (anti-aliasing)
CAM_HFOV = math.radians(100.0)
CAM_HEIGHT = 1.20          # above ground [m]
CAM_PITCH = math.radians(15.0)   # pitched down
CAM_X = 1.00               # ahead of rear axle [m]
MAX_RANGE = 60.0           # rays beyond this are sky/haze [m]
HAZE_DIST = 45.0           # exponential haze scale [m]

# ---------------- expert / navigation ---------------------------------------
LOOKAHEAD_MIN = 3.0
LOOKAHEAD_GAIN = 0.4       # Ld = LOOKAHEAD_MIN + LOOKAHEAD_GAIN * v
CMD_BEFORE = 18.0          # issue LEFT/RIGHT this many metres before the fork
CMD_AFTER = 8.0            # ... and keep it this long after the fork
# command indices
FOLLOW, LEFT, RIGHT = 0, 1, 2
