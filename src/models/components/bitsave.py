from typing import Tuple

import torch
from torch import Tensor, nn

from src.models.components.conv import conv_layer, t_conv_layer
from src.models.components.rfdb import RFDB


class BitSave(nn.Module):
    def __init__(self, hiddden_channels: int, num_blocks: int):
        super().__init__()

        self.hiddden_channels = hiddden_channels

        self.y_input_conv = conv_layer(1, hiddden_channels, kernel_size=3, stride=2)
        self.uv_input_conv = conv_layer(2, hiddden_channels, kernel_size=3)
        self.blocks = nn.ModuleList([RFDB(hiddden_channels * 2) for _ in range(num_blocks)])
        self.blocks_out_conv = conv_layer(
            hiddden_channels * 2 * num_blocks, hiddden_channels * 2, kernel_size=1
        )
        self.activation = nn.LeakyReLU()
        self.y_output_conv = t_conv_layer(hiddden_channels, 1, kernel_size=3, stride=2)
        self.uv_output_conv = t_conv_layer(hiddden_channels, 2, kernel_size=3)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        y, uv = x
        y = self.y_input_conv(y)
        uv = self.uv_input_conv(uv)
        block_input = torch.cat([y, uv], dim=1)
        block_outputs = []
        for block in self.blocks:
            block_input = block(block_input)
            block_outputs.append(block_input)

        out = torch.cat(block_outputs, dim=1)
        out = self.activation(self.blocks_out_conv(out))
        y, uv = torch.split(out, self.hiddden_channels, dim=1)
        y = self.y_output_conv(y)
        uv = self.uv_output_conv(uv)
        return y, uv
