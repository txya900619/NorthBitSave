from io import BytesIO
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision import transforms
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg


class JpegCompression:
    def __init__(self):
        pass

    def __call__(self, img: Tensor) -> Tensor:
        # c = [25, 18, 15, 10, 7]
        qualities = [25, 18, 15]
        index = np.random.randint(0, len(qualities))

        quality = qualities[index]

        return decode_jpeg(encode_jpeg(img, quality=quality), mode=ImageReadMode.RGB)


class GaussianNoise:
    def __init__(self):
        pass

    def __call__(self, img: Tensor) -> Tensor:
        # c = np.random.uniform(.08, .38)
        min = 0.08

        std = np.random.uniform(min, min + 0.03)
        img = torch.clip(img + torch.normal(0, std, size=img.shape), min=0, max=1)
        return img


class ShotNoise:
    def __init__(self):
        pass

    def __call__(self, img: Tensor) -> Tensor:
        # c = np.random.uniform(3, 60)
        min = 3
        scale = np.random.uniform(min, min + 7)
        img = torch.clip(torch.poisson(img * scale) / float(scale), min=0, max=1)
        return img


class GaussianBlur:
    def __init__(self):
        pass

    def __call__(self, img: Tensor) -> Tensor:
        # kernel = [(31,31)] prev 1 level only
        kernel = (31, 31)
        sigmas = [0.5, 1, 2]
        index = np.random.randint(0, len(sigmas))

        sigma = sigmas[index]
        return transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)(img)


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        mode = np.random.choice(
            [transforms.InterpolationMode.BICUBIC, transforms.InterpolationMode.BILINEAR, "AREA"]
        )
        if mode == "AREA":
            return F.interpolate(img.unsqueeze(0), size=self.size, mode="area").squeeze(0)

        else:
            img = transforms.ConvertImageDtype(torch.uint8)(img)
            img = transforms.Resize(size=self.size, interpolation=mode)(img)
            img = transforms.ConvertImageDtype(torch.float)(img)
            return img


class RandomCrop:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        return transforms.RandomCrop(size=self.size)(img)
