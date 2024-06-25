from typing import Tuple

import torch
from torch import Tensor

YCBCR_WEIGHTS = {"ITU-R_BT.709": (0.2126, 0.7152, 0.0722)}


def rgb_to_ycbcr420(image: Tensor) -> Tuple[Tensor, Tensor]:
    """Input is 3xhxw RGB float numpy array, in the range of [0, 1] output is y: 1xhxw, uv:

    2x(h/2)x(w/x), in the range of [0, 1]
    """
    rgb = image
    c, h, w = rgb.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    r, g, b = torch.split(rgb, 1, dim=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    # to 420
    cb = cb.reshape((1, h // 2, 2, w // 2, 2)).mean((-1, -3))
    cr = cr.reshape((1, h // 2, 2, w // 2, 2)).mean((-1, -3))
    uv = torch.cat((cb, cr), dim=0)

    y = torch.clip(y, min=0.0, max=1.0)
    uv = torch.clip(uv, min=0.0, max=1.0)

    return y, uv


def rgb_to_ycbcr(rgb):
    """
    input is 3xhxw or bx3xhxw RGB float numpy array, in the range of [0, 1]
    output is yuv: 3xhxw or bx3xhxw, in the range of [0, 1]
    """
    if rgb.shape[-3] != 3:
        raise Exception("Image channel should be 3")

    r, g, b = rgb.chunk(3, -3)

    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    ycbcr = torch.cat((y, cb, cr), dim=-3)
    ycbcr = torch.clip(ycbcr, min=0.0, max=1.0)

    return ycbcr


def ycbcr_to_rgb(y: Tensor, uv: Tensor) -> Tensor:
    """
    y is 1xhxw or bx1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2xhxw or bx2xhxw UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    """
    if len(y.shape) != len(uv.shape):
        raise Exception("y and uv shape length should be the same")

    shape_length = len(y.shape)
    if shape_length == 3:
        channel_index = 0
    elif shape_length == 4:
        channel_index = 1
    else:
        raise Exception("y shape should be 1xhxw or bx1xhxw")

    if uv.shape[channel_index] != 2:
        raise Exception("uv channel should be 2")

    cb, cr = torch.split(uv, 1, dim=channel_index)

    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=channel_index)
    rgb = torch.clip(rgb, min=0.0, max=1.0)
    return rgb
