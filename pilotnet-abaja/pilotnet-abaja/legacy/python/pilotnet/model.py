"""PilotNet (Bojarski et al., 2016) with optional command-conditioned branches
(Codevilla et al., 2018).

Input  : RGB image, float32, range 0..255, shape (N, 3, 66, 200).
         Colour conversion (RGB -> YUV) and normalisation are *inside* the
         network, so the exported ONNX model takes raw pixels. This removes the
         most common train/deploy mismatch (forgetting to convert or normalise
         at inference time, e.g. in MATLAB).
Output : (N, n_branches) normalised steering targets. With n_branches == 1 the
         network is exactly PilotNet. With n_branches == 3 the heads are
         [FOLLOW, LEFT, RIGHT]; the navigation system picks one head at run time.

Layer sizes (66x200 input):
    conv1 5x5/2  24 -> 31x98
    conv2 5x5/2  36 -> 14x47
    conv3 5x5/2  48 ->  5x22
    conv4 3x3/1  64 ->  3x20
    conv5 3x3/1  64 ->  1x18   -> flatten 1152
    fc 100 -> 50 -> 10 -> 1
    ~252 k parameters for one branch, matching the "about 250 thousand
    parameters" reported by NVIDIA. (The paper's figure labels the flatten
    layer "1164"; 64*1*18 = 1152, and a 1164-unit dense layer would add
    ~1.3 M parameters, contradicting the stated total, so we use 1152.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

COMMANDS = ("FOLLOW", "LEFT", "RIGHT")


class RGB2YUVNorm(nn.Module):
    """Fixed RGB[0,255] -> YUV (BT.601 analogue) -> roughly [-1, 1].

    Y = 0.299 R + 0.587 G + 0.114 B
    U = -0.14713 R - 0.28886 G + 0.436 B + 128
    V = 0.615 R - 0.51499 G - 0.10001 B + 128
    out = YUV / 127.5 - 1
    Implemented as a frozen 1x1 convolution so it exports cleanly to ONNX.
    """

    def __init__(self) -> None:
        super().__init__()
        m = torch.tensor([[0.299, 0.587, 0.114],
                          [-0.14713, -0.28886, 0.436],
                          [0.615, -0.51499, -0.10001]], dtype=torch.float32)
        b = torch.tensor([0.0, 128.0, 128.0], dtype=torch.float32)
        self.register_buffer("weight", m.view(3, 3, 1, 1))
        self.register_buffer("bias", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias) / 127.5 - 1.0


def _act(name: str) -> nn.Module:
    return {"elu": nn.ELU(), "relu": nn.ReLU(), "gelu": nn.GELU()}[name]


class PilotNet(nn.Module):
    def __init__(self, n_branches: int = 1, dropout: float = 0.2,
                 act: str = "elu", input_hw: tuple[int, int] = (66, 200),
                 batchnorm: bool = False) -> None:
        super().__init__()
        self.n_branches = n_branches
        self.input_hw = input_hw
        self.norm = RGB2YUVNorm()

        def conv(cin, cout, k, s):
            layers = [nn.Conv2d(cin, cout, k, s)]
            if batchnorm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(_act(act))
            return nn.Sequential(*layers)

        self.convs = nn.ModuleList([
            conv(3, 24, 5, 2), conv(24, 36, 5, 2), conv(36, 48, 5, 2),
            conv(48, 64, 3, 1), conv(64, 64, 3, 1),
        ])
        with torch.no_grad():
            f = torch.zeros(1, 3, *input_hw)
            for c in self.convs:
                f = c(f)
        self.flat_dim = int(f.numel())

        def head():
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.flat_dim, 100), _act(act), nn.Dropout(dropout),
                nn.Linear(100, 50), _act(act),
                nn.Linear(50, 10), _act(act),
                nn.Linear(10, 1))

        self.heads = nn.ModuleList([head() for _ in range(n_branches)])

    def trunk(self, x: torch.Tensor, keep: bool = False):
        x = self.norm(x)
        feats = []
        for c in self.convs:
            x = c(x)
            if keep:
                feats.append(x)
        return (x, feats) if keep else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.trunk(x).flatten(1)
        return torch.cat([h(f) for h in self.heads], dim=1)


def select_branch(out: torch.Tensor, cmd: torch.Tensor | None) -> torch.Tensor:
    """Pick each sample's head given its command index (0=FOLLOW,1=LEFT,2=RIGHT)."""
    if out.shape[1] == 1 or cmd is None:
        return out[:, 0]
    return out.gather(1, cmd.long().view(-1, 1)).squeeze(1)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for nb in (1, 3):
        net = PilotNet(n_branches=nb)
        y = net(torch.rand(2, 3, 66, 200) * 255)
        print(f"branches={nb} flat={net.flat_dim} params={count_parameters(net):,} out={tuple(y.shape)}")
