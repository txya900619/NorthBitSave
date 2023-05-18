from typing import Tuple

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def rgb_to_ycbcr420(image: Tensor) -> Tuple[Tensor, Tensor]:
    """input is 3xhxw RGB float numpy array, in the range of [0, 1] output is y: 1xhxw, uv:

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


# def ycbcr420_to_rgb(y: np.ndarray, uv: np.ndarray, order=1)->np.ndarray:
#     '''
#     y is 1xhxw Y float numpy array, in the range of [0, 1]
#     uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
#     order: 0 nearest neighbor, 1: binear (default)
#     return value is 3xhxw RGB float numpy array, in the range of [0, 1]
#     '''
#     uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
#     cb = uv[0:1, :, :]
#     cr = uv[1:2, :, :]
#     Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
#     r = y + (2 - 2 * Kr) * (cr - 0.5)
#     b = y + (2 - 2 * Kb) * (cb - 0.5)
#     g = (y - Kr * r - Kb * b) / Kg
#     rgb = np.concatenate((r, g, b), axis=0)
#     rgb = np.clip(rgb, 0., 1.)
#     return rgb
