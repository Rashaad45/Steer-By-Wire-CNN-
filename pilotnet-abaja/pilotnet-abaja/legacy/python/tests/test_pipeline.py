"""Fast correctness tests:  pytest -q tests/"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilotnet.augment import flip  # noqa: E402
from pilotnet.data import block_split  # noqa: E402
from pilotnet.metrics import autonomy, regression_metrics  # noqa: E402
from pilotnet.model import PilotNet, count_parameters, select_branch  # noqa: E402
from pilotnet.vbp import visual_backprop  # noqa: E402
from sim import config as C  # noqa: E402
from sim.course import astar, build_course, build_route  # noqa: E402
from sim.render import camera_precompute  # noqa: E402
from sim.vehicle import bicycle_step  # noqa: E402


def test_pilotnet_shapes_and_params():
    m = PilotNet()
    assert m.flat_dim == 1152
    assert 250_000 < count_parameters(m) < 255_000          # "about 250 thousand parameters"
    assert m(torch.rand(4, 3, 66, 200) * 255).shape == (4, 1)
    m3 = PilotNet(n_branches=3)
    out = m3(torch.rand(4, 3, 66, 200) * 255)
    sel = select_branch(out, torch.tensor([0, 1, 2, 1]))
    assert torch.allclose(sel, out[torch.arange(4), torch.tensor([0, 1, 2, 1])])


def test_yuv_layer_matches_formula():
    m = PilotNet()
    x = torch.tensor([255.0, 0.0, 0.0]).view(1, 3, 1, 1)        # pure red
    yuv = (m.norm(x) + 1) * 127.5
    assert torch.allclose(yuv.flatten(), torch.tensor([0.299 * 255, -0.14713 * 255 + 128, 0.615 * 255 + 128]),
                          atol=1e-3)


def test_vbp_mask_shape_and_range():
    m = PilotNet()
    mask = visual_backprop(m, torch.rand(2, 3, 66, 200) * 255)
    assert mask.shape == (2, 66, 200)
    assert float(mask.min()) >= 0 and float(mask.max()) <= 1 + 1e-6


def test_flip_negates_and_swaps_commands():
    img = np.arange(66 * 200 * 3, dtype=np.uint8).reshape(66, 200, 3)
    out, t, c = flip(img, 0.3, 1)
    assert t == -0.3 and c == 2 and np.array_equal(out[:, 0], img[:, -1])


def test_block_split_has_no_overlap_and_is_blocky():
    sp = block_split(10_000, 500, 0.1, 0.1)
    a, b, c = map(set, (sp["train"], sp["val"], sp["test"]))
    assert not (a & b or a & c or b & c) and len(a | b | c) == 10_000
    assert all((i // 500) == (sp["test"][0] // 500) for i in sp["test"][:500])


def test_astar_and_commands():
    course = build_course()
    names = [course.edges[e]["to"] for e in astar(course, "S", "B")]
    assert names == ["J1", "B"]
    r = build_route(course, ["S", "A", "S", "B", "S"])
    assert [f["cmd"] for f in r.forks] == [C.LEFT, C.RIGHT]
    assert np.all(np.diff(r.s) > 0)


def test_bicycle_turning_radius():
    st = [0.0, 0.0, 0.0, C.MAX_STEER]
    for _ in range(int(2 * math.pi * (C.WHEELBASE / math.tan(C.MAX_STEER)) / C.SPEED / C.DT)):
        st = bicycle_step(st, C.MAX_STEER, C.SPEED)
    assert math.hypot(st[0], st[1]) < 0.5                    # closed a full circle


def test_camera_geometry():
    cam = camera_precompute()
    # bottom-centre pixel looks at the ground just ahead of the camera
    pf = cam["pf"][-1, cam["pf"].shape[1] // 2]
    pitch_bottom = C.CAM_PITCH + math.atan((C.IMG_H / 2) / ((C.IMG_W / 2) / math.tan(C.CAM_HFOV / 2)))
    assert pf == pytest.approx(C.CAM_X + C.CAM_HEIGHT / math.tan(pitch_bottom), rel=0.03)
    assert not cam["valid"][0].any()                          # top row is sky


def test_metrics():
    r = regression_metrics([0, 1, 2], [0, 1, 2])
    assert r["rmse"] == 0 and r["r2"] == pytest.approx(1)
    assert autonomy(10, 600) == pytest.approx(90.0)           # NVIDIA's worked example
