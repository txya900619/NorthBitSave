import torch
from torch import Tensor, nn

from src.models.components.conv import conv_layer
from src.models.components.rfdb import RFDB


class BitSave(nn.Module):
    def __init__(self, hiddden_channels: int, num_blocks: int):
        super().__init__()

        self.hiddden_channels = hiddden_channels

        self.y_input_conv = conv_layer(1, hiddden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([RFDB(hiddden_channels) for _ in range(num_blocks)])
        self.blocks_out_conv = conv_layer(
            hiddden_channels * num_blocks, hiddden_channels, kernel_size=1
        )
        self.activation = nn.LeakyReLU()
        self.y_output_conv = conv_layer(hiddden_channels, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        y = x
        y = self.y_input_conv(y)
        block_input = y
        block_outputs = []
        for block in self.blocks:
            block_input = block(block_input)
            block_outputs.append(block_input)

        out = torch.cat(block_outputs, dim=1)
        out = self.activation(self.blocks_out_conv(out))
        y = self.y_output_conv(out)
        return y
