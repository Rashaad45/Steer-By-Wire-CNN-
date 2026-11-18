"""VisualBackProp (Bojarski et al., ICRA 2018): which input pixels drive the
steering prediction.

1. Average every conv layer's feature maps over channels.
2. Start at the deepest averaged map; repeatedly up-scale it to the previous
   layer's size with a transposed convolution whose kernel is all ones (same
   kernel size / stride as the forward conv), and multiply point-wise with that
   layer's averaged map.
3. Up-scale once more to the input resolution and normalise to [0, 1].
Cost is below one forward pass, so it can run online.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def visual_backprop(model, x: torch.Tensor) -> torch.Tensor:
    """x: (N,3,H,W) float RGB 0..255. Returns masks (N,H,W) in [0,1]."""
    model.eval()
    _, feats = model.trunk(x, keep=True)
    convs = [c[0] for c in model.convs]                        # nn.Conv2d of each block
    sizes = [x.shape[-2:]] + [f.shape[-2:] for f in feats]
    m = feats[-1].mean(1, keepdim=True)
    for li in range(len(feats) - 1, -1, -1):
        conv = convs[li]
        k, s = conv.kernel_size, conv.stride
        tgt = sizes[li]
        out_h = (m.shape[-2] - 1) * s[0] + k[0]
        out_w = (m.shape[-1] - 1) * s[1] + k[1]
        op = (tgt[0] - out_h, tgt[1] - out_w)
        w = torch.ones(1, 1, *k, device=x.device)
        m = F.conv_transpose2d(m, w, stride=s, output_padding=op)
        if li > 0:
            m = m * feats[li - 1].mean(1, keepdim=True)
    m = m[:, 0]
    mn = m.flatten(1).min(1)[0].view(-1, 1, 1)
    mx = m.flatten(1).max(1)[0].view(-1, 1, 1)
    return (m - mn) / (mx - mn + 1e-8)
