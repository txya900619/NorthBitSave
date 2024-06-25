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
"""Encode-Decode library of functions that emulate intra compression scenarios."""

import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.models.components.JPEG_proxy.jpeg_proxy import JpegProxy, differentiable_round


def _encode_decode_with_jpeg(
    input_images: np.ndarray,
    qstep: np.float32,
    one_channel_at_a_time: bool = False,
    use_420: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compress-decompress with actual jpeg with fixed qstep.

    Args:
      input_images: Array of shape [b, n, m, c] where b is batch size, n x m is
        the image size, and c is the number of channels.
      qstep: float that determines the step-size of the scalar quantizer.
      one_channel_at_a_time: True if each channel should be encoded independently
        as a grayscale image.
      use_420: True when desired subsmapling is 4:2:0. False when 4:4:4.

    Returns:
      decoded: Array of same size as input_images containing the
        quantized-dequantized version of the input_images.
      rate: Array of size b that contains the total number of bits needed to
        encode the input_images into decoded.
    """

    assert input_images.ndim == 4
    decoded = np.zeros_like(input_images)
    rate = np.zeros(input_images.shape[0])
    # Jpeg needs byte qsteps
    jpeg_qstep = np.clip(np.rint(qstep).astype(int), 0, 255)
    qtable = [jpeg_qstep] * 64

    def run_jpeg(input_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = Image.fromarray(np.rint(np.clip(input_image, 0, 255)).astype(np.uint8))
        buf = io.BytesIO()
        img.save(
            buf,
            format="jpeg",
            quality=100,
            optimize=True,
            qtables=[qtable, qtable, qtable],
            subsampling="4:2:0" if use_420 else "4:4:4",
        )
        decoded = np.array(Image.open(buf))
        rate = np.array(8 * len(buf.getbuffer()))
        return decoded, rate

    for index in range(input_images.shape[0]):
        if not one_channel_at_a_time:
            decoded[index], rate[index] = run_jpeg(input_images[index])
        else:
            # Run each channel separately through jpeg as a grayscale image
            # (Image.mode = 'L'.) Useful when RGB <-> YUV conversions need to be
            # skipped.
            for channel in range(input_images.shape[-1]):
                decoded[index, ..., channel], channel_rate = run_jpeg(
                    input_images[index, ..., channel]
                )
                rate[index] += channel_rate

    return decoded.astype(np.float32), rate.astype(np.float32)


class EncodeDecodeIntra(nn.Module):
    """A class with methods for basic intra compression emulation."""

    def __init__(
        self,
        use_jpeg_rate_model: bool = True,
        qstep_init: float = 1.0,
        train_qstep: bool = True,
        min_qstep: float = 0.0,
        jpeg_clip_to_0_255: bool = True,
        downsample_chroma: bool = False,
    ):
        """Constructor.

        Args:
          use_jpeg_rate_model: True for JPEG-specific rate model, False for
            Gaussian-distribution-based rate model.
          qstep_init: float that determines initial value for the step-size of the
            scalar quantizer.
          train_qstep: Whether qstep should be trained. When False the class will
            use qstep_init or any qsteps provided in the call(). The latter is
            useful when the same module is used in video with different qsteps for
            INTRA and INTER.
          min_qstep: Minimum value which qstep should be greater than. Set to 1 to
            reflect for some practical codecs that cannot go below integer values.
          jpeg_clip_to_0_255: True if jpeg proxy should clip the final output to [0,
            255]. Set to False when handling INTER frames.
        """
        super().__init__()

        self.train_qstep = train_qstep
        if self.train_qstep:
            self.qstep = nn.Parameter(torch.tensor(qstep_init, dtype=torch.float32))
        else:
            self.qstep = torch.tensor(qstep_init, dtype=torch.float32)

        self.min_qstep = torch.tensor(min_qstep, dtype=torch.float32)

        self.clip_to_0_255 = jpeg_clip_to_0_255

        def _quantizer_fn(x: torch.Tensor) -> torch.Tensor:
            """Implements quantize-dequantize with the trainable qstep."""
            positive_qstep = self._positive_qstep()
            return differentiable_round(x / positive_qstep) * positive_qstep

        self._jpeg_quantizer_fn = _quantizer_fn

        self.use_jpeg_rate_model = use_jpeg_rate_model
        self.run_jpeg_one_channel_at_a_time = True
        self.downsample_chroma = downsample_chroma

        self._jpeg_layer = JpegProxy(
            downsample_chroma=downsample_chroma,
            clip_to_0_255=self.clip_to_0_255,
        )

        # Workaround thread-unsafe PIL library by calling init in main thread.
        Image.init()

    def _positive_qstep(self) -> torch.Tensor:
        return F.elu(self.qstep, alpha=0.01) + self.min_qstep

    def get_qstep(self) -> torch.Tensor:
        return self._positive_qstep()

    def _rate_proxy_gaussian(self, inputs: torch.Tensor, axis: List[int]) -> torch.Tensor:
        """Calculates entropy assuming a Gaussian distribution and high-res quantization.

        Args:
          inputs: Tensor of shape [b, n1, ...].
          axis: Axis of random variable realizations, e.g., with inputs b x n1 x n2
            and axis=[1] then there are n2 Gaussian variables with potentially
            different distributions, each with samples along axis=[1].

        Returns:
          rate: Tensor of shape [b] that estimates the total number of bits needed
            to represent the values quantized with self.qstep.
        """
        assert inputs.dim() >= np.max(np.abs(axis))
        deviations = torch.std(inputs, dim=axis, keepdim=False)
        assert deviations.shape[0] == inputs.shape[0]

        hires_entropy = F.relu(
            torch.log(deviations / self._positive_qstep() + torch.finfo(torch.float32).eps)
            + 0.5 * np.log(2 * np.pi * np.exp(1))
        )

        # Sum the entropies for total rate
        return (
            hires_entropy.reshape(inputs.shape[0], -1).sum(dim=1)
            * torch.tensor(inputs.shape, dtype=torch.float32)[axis].prod()
            / np.log(2)
        )

    def _rate_proxy_jpeg(
        self,
        three_channel_inputs: torch.Tensor,
        dequantized_dct_coeffs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculates a rate proxy based on a Jpeg-specific rate model."""

        def calculate_non_zeros(
            dct_coeffs: Dict[str, torch.Tensor], qstep: torch.Tensor
        ) -> torch.Tensor:
            num_nonzeros = torch.zeros(three_channel_inputs.shape[0])
            for k in dct_coeffs:
                num_nonzeros += (
                    (1 + (dct_coeffs[k] / qstep).abs())
                    .log()
                    .view(three_channel_inputs.shape[0], -1)
                    .sum(dim=1)
                )

            return num_nonzeros

        def encode_decode_inputs_with_jpeg() -> Tuple[torch.Tensor, torch.Tensor]:
            """Encodes then decodes the three_channel_inputs using actual jpeg."""
            jpeg_decoded, jpeg_rate = _encode_decode_with_jpeg(
                three_channel_inputs.cpu().numpy(),
                self._positive_qstep().cpu().numpy(),
                self.run_jpeg_one_channel_at_a_time,
                self.downsample_chroma,
            )

            # jpeg_decoded.set_shape(three_channel_inputs.shape)
            # jpeg_rate.set_shape(three_channel_inputs.shape[0])

            jpeg_decoded = torch.from_numpy(jpeg_decoded).to(three_channel_inputs.device)
            jpeg_rate = torch.from_numpy(jpeg_rate).to(three_channel_inputs.device)
            return jpeg_decoded, jpeg_rate

        ###########################################################################
        # Jpeg-specific model fits rate using number of nonzero dct coefficients.
        # For details see:
        #  Z. He and S. K. Mitra, "A unified rate-distortion analysis framework for
        #  transform coding," in IEEE Transactions on Circuits and Systems for Video
        #  Technology, vol. 11, no. 12, pp. 1221-1236, Dec. 2001.
        #
        # Generate (rate, num_nonzero) pairs, fit a weight as
        # rate ~= weight * num_nonzero, return rate approximation as weight *
        # num_nonzero.
        ###########################################################################

        # First pair using current qstep. (May consider fitting to a batch instead.)
        num_nonzero_dct_coeffs = calculate_non_zeros(
            dequantized_dct_coeffs, self._positive_qstep()
        )
        _, jpeg_rate = encode_decode_inputs_with_jpeg()

        nonzero_times_rate = num_nonzero_dct_coeffs * jpeg_rate
        nonzero_times_nonzero = num_nonzero_dct_coeffs * num_nonzero_dct_coeffs

        # This is in effect a fit within the main training using a different loss
        # function whose solution is known. Stop the gradients that may confuse
        # the optimizer.
        line_weights = (nonzero_times_rate / (nonzero_times_nonzero + 1)).detach()

        return num_nonzero_dct_coeffs * line_weights

    def _encode_decode_jpeg(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes then decodes the input using JPEG.

        Args:
          inputs: Tensor of shape [b, n, m, c] where b is batch size, n x m is the
            image size, and c is the number of channels (c <= 3).

        Returns:
          outputs: Tensor of same shape as inputs containing the
            quantized-dequantized version of the inputs.
          rate: Tensor of shape [b] that estimates the total number of bits needed
            to encode the input into output.
        """
        if inputs.dim() != 4:
            raise ValueError("inputs must have rank 4.")
        if inputs.shape[-1] > 3:
            raise ValueError("jpeg layer can handle up to 3 channels.")

        # Zero-pad to three channels as needed for the jpeg layer.
        pad_dim = 3 - inputs.shape[-1]
        if pad_dim:
            three_channel_inputs = F.pad(inputs, (0, pad_dim), mode="constant", value=0)
        else:
            three_channel_inputs = inputs

        # Emulate integer inputs needed when using an actual intra codec.
        three_channel_inputs = differentiable_round(three_channel_inputs)

        # JPEG quantize-dequantize.
        dequantized_three_channels, dequantized_dct_coeffs = self._jpeg_layer(
            three_channel_inputs, self._jpeg_quantizer_fn
        )

        # May consider adding self._rounding_fn(dequantized_three_channels) if
        # desired.

        # Remove padding.
        if pad_dim:
            dequantized = dequantized_three_channels[:, :, :, : inputs.shape[-1]]
        else:
            dequantized = dequantized_three_channels

        if self.use_jpeg_rate_model:
            rate = self._rate_proxy_jpeg(three_channel_inputs, dequantized_dct_coeffs)
        else:
            gauss_rate = torch.zeros(inputs.shape[0])
            for k in dequantized_dct_coeffs:
                gauss_rate += self._rate_proxy_gaussian(dequantized_dct_coeffs[k], axis=[1])
            rate = gauss_rate

        return dequantized, rate

    def forward(
        self, inputs: torch.Tensor, input_qstep: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes then decodes the input.

        Args:
          inputs: Tensor of shape [b, n, m, c] where b is batch size, n x m is the
            image size, and c is the number of channels. (0~255)
          input_qstep: qstep to use when self.qstep is not trained.

        Returns:
          outputs: Tensor of same size as inputs containing the
            quantized-dequantized version of the inputs.
          rate: Tensor of size b that estimates the total number of bits needed to
            encode the input into output.
        """
        if inputs.dim() != 4:
            raise ValueError("inputs must have dim 4.")

        if not self.train_qstep and input_qstep is not None:
            self.qstep = input_qstep

        if inputs.shape[-1] <= 3:
            return self._encode_decode_jpeg(inputs)

        # JPEG layer handles at most three channels. Run three channels at a time.
        # (i) Run first three channels to initialize the return tensors.
        size = np.array(inputs.shape, dtype=np.int32)
        limit = size[-1]
        size[-1] = 3
        begin = np.zeros_like(size, dtype=np.int32)
        dequantized, rate = dequantized, rate = self._encode_decode_jpeg(
            inputs[:, :, :, begin[-1] : size[-1]].contiguous()
        )

        # (ii) Run three channels at a time and update.
        for i in range(3, limit, 3):
            begin[-1] = i
            size[-1] = np.minimum(limit - begin[-1], 3)
            dequantized_loop, rate_loop = self._encode_decode_jpeg(
                inputs[:, :, :, begin[-1] : begin[-1] + size[-1]].contiguous()
            )

            # Update the return variables
            dequantized = torch.cat([dequantized, dequantized_loop], dim=3)
            rate += rate_loop
        return dequantized, rate
