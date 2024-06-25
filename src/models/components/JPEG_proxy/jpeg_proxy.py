# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Jpeg proxy that emulates intra compression with jpeg."""

import itertools
from typing import Dict, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import InterpolationMode, resize

# def torch_extract_patches(x: torch.Tensor, patch_height: int, patch_width: int):
#     # x = x.unsqueeze(0)

#     patches = F.unfold(
#         x, (patch_height, patch_width), stride=(patch_height, patch_width)
#     )
#     patches = patches.reshape(x.size(0), x.size(1), patch_height, patch_width, -1)
#     patches = patches.permute(0, 4, 2, 3, 1).reshape(
#         x.size(0),
#         x.size(2) // patch_height,
#         x.size(3) // patch_width,
#         x.size(1) * patch_height * patch_width,
#     )
#     return patches


def differentiable_round(x: torch.Tensor) -> torch.Tensor:
    return x + x.round().detach() - x.detach()


class JpegProxy:
    """Differentiable JPEG-like layer.

    Simplified for sandwich use.
    """

    def __init__(
        self,
        downsample_chroma: bool,
        clip_to_0_255: bool,
        dct_size: int = 8,
    ):  # pytype: disable=annotation-type-mismatch
        """Constructor.

        Module that accomplishes the JPEG-like pipeline:
          (rgb-to-yuv) -> (chroma-downsampling) -> forward-DCT -> quantization ->
          dequantization -> inverse-DCT -> (chroma-upsampling) -> (yuv-to-rgb)

        Args:
          downsample_chroma: Whether to downsample chroma channels. Downsampling is
            bilinear, upsampling is nearest neighbor.
          dct_size: Size of the core 1D DCT transform.
          clip_to_0_255: True if final output should be clipped to [0, 255].
        """
        self.downsample_chroma = downsample_chroma
        self.luma_quantization_table = torch.ones(dct_size**2)
        self.chroma_quantization_table = torch.ones(dct_size**2)
        self.clip_to_0_255 = clip_to_0_255

        self.dct_size = dct_size
        self.dct_2d_mat = self._construct_dct_2d(self.dct_size)

    def _construct_dct_2d(self, dct_size: int) -> torch.Tensor:
        """Returns a matrix containing the 2D DCT basis in its columns."""
        # 1D basis, unit norm, in columns.
        # https://en.wikipedia.org/wiki/Discrete_cosine_transform (DCTII)
        dct_1d_mat = np.zeros((dct_size, dct_size), dtype=np.float32)

        for i, j in itertools.product(range(dct_size), repeat=2):
            dct_1d_mat[i, j] = np.cos((2 * i + 1) * j * np.pi / (2 * dct_size))

        # Scale for unit norm.
        dct_1d_mat *= np.sqrt(2 / dct_size)
        dct_1d_mat[:, 0] *= 1 / np.sqrt(2)

        block_size = dct_size**2

        # 2D basis, unit norm, in columns.
        dct_2d_mat = np.zeros((block_size, block_size), dtype=np.float32)

        for i in range(block_size):
            dct_2d_mat[:, i] = np.reshape(
                np.outer(dct_1d_mat[:, i // dct_size], dct_1d_mat[:, i % dct_size]),
                [-1],
            )
        assert np.sum(np.abs(np.matmul(dct_2d_mat, dct_2d_mat.T) - np.eye(block_size))) <= 1e-3
        return torch.from_numpy(dct_2d_mat)

    def _forward_dct_2d(self, image_channel: torch.Tensor) -> torch.Tensor:
        """Returns the 2D DCT coefficients of the input image channel."""
        image_patches = image_channel.unfold(1, self.dct_size, self.dct_size).unfold(
            2, self.dct_size, self.dct_size
        )
        image_patches = image_patches.permute(0, 1, 2, 4, 5, 3).contiguous()
        image_patches = image_patches.view(
            image_patches.size(0), image_patches.size(1), image_patches.size(2), -1
        )
        offset = 128
        # return coefficients, with shape:
        #   [batch_size, height / dct_size, width / dct_size, dct_size * dct_size]
        return torch.matmul(image_patches - offset, self.dct_2d_mat)

    def _inverse_dct_2d(self, dct_coeffs: torch.Tensor) -> torch.Tensor:
        """Returns the image that is the inverse transform of the coeffs."""
        offset = 128
        # channel is pixel domain, with shape:
        #   [batch_size, height / dct_size, width / dct_size, dct_size * dct_size]
        channel = torch.matmul(dct_coeffs, self.dct_2d_mat.T) + offset
        # after reshape:
        #   [batch_size, height / dct_size, width / dct_size, dct_size, dct_size]
        channel = channel.reshape(*channel.shape[0:-1], self.dct_size, self.dct_size)
        # return channel with shape: [batch_size, 1, height, width]
        return channel.permute(0, 1, 3, 2, 4).reshape(
            channel.shape[0],
            -1,
            channel.shape[1] * self.dct_size,
            channel.shape[2] * self.dct_size,
        )

    def __call__(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compresses and decompresses input an image similar to JPEG.

        Args:
          image: Input image of size [batch_size, height, width, 3]. (0~255)
          rounding_fn: Rounding function used in the quantization. One can establish
            quantization via two ways: (i) Establish quantization tables then use a
            rounding function that rounds to the closest integer. (ii) Set
            quantization tables to 1 and implement quantization in the rounding_fn.
            This is the preferred mode when the qstep(s) used during quantization
            are optimized and maintained elsewhere. See EncodeDecodeIntra in
            encode_decode_intra_lib.py.

        Returns:
          Compressed image of the same size and type as input. (0~255)
          Dictionary containing the quantized DCT coefficients of Y, U and V
            channels. Each channel has shape:
              [batch_size, h / dct_size, w / dct_size, dct_size * dct_size],
            where (h, w) are the suitably padded height and width of that channel.
            For images with dimensions that are not a multiple of dct_size,
            symmetric extension is used.
        """
        assert image.dim() >= 4
        assert image.shape[-1] == 3
        height, width = image.shape[1:3]

        # Pad to a multiple of dct_size (or 2 * dct_size if downsampling chroma.)
        pad_multiple = 2 * self.dct_size if self.downsample_chroma else self.dct_size
        pad_height = ((height - 1) // pad_multiple + 1) * pad_multiple - height
        pad_width = ((width - 1) // pad_multiple + 1) * pad_multiple - width
        if pad_height or pad_width:
            image = F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")

        # Encode-Decode

        downsample = [False, self.downsample_chroma, self.downsample_chroma]
        dct_keys = ["y", "u", "v"]
        padded_height = height + pad_height
        padded_width = width + pad_width
        decoded_image = []

        quantized_dct_coeffs = {}
        for ch in range(3):
            if downsample[ch]:
                channel = resize(
                    image[..., ch : ch + 1],
                    [padded_height // 2, padded_width // 2],
                )
            else:
                channel = image[..., ch : ch + 1]

            # Forward DCT.
            coeffs = self._forward_dct_2d(channel)

            # Quantized DCT coefficients.
            quantization_table = (
                self.luma_quantization_table if ch == 0 else self.chroma_quantization_table
            )
            quantized_dct_coeffs[dct_keys[ch]] = differentiable_round(coeffs / quantization_table)

            # Dequantized DCT coefficients.
            dequantized = quantized_dct_coeffs[dct_keys[ch]] * quantization_table

            # Inverse DCT.
            channel = self._inverse_dct_2d(dequantized)

            if downsample[ch]:
                channel = resize(
                    channel,
                    [padded_height, padded_width],
                    InterpolationMode.NEAREST,
                )
            decoded_image = channel if ch == 0 else torch.cat([decoded_image, channel], dim=-1)

        # Back to original size.
        decoded_image = decoded_image[:, :height, :width, :]
        if self.clip_to_0_255:
            decoded_image = torch.clip(decoded_image, 0.0, 255.0)
        return decoded_image, quantized_dct_coeffs
