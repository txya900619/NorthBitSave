import torch
from torch import Tensor, nn

from src.models.components.AsConvSR.Assembled_conv import AssembledBlock


class AsConvSR(nn.Module):
    def __init__(self, hidden_channels=32, E=3, temperature=30, ratio=4, global_residual=True):
        super().__init__()

        self.global_residual = global_residual

        self.pixelUnShuffle = nn.PixelUnshuffle(2)
        self.conv1 = nn.Conv2d(4, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.assemble = AssembledBlock(
            hidden_channels,
            hidden_channels,
            E=E,
            kernel_size=3,
            stride=1,
            padding=1,
            temperature=temperature,
            ratio=ratio,
        )
        self.assemble2 = AssembledBlock(
            hidden_channels,
            hidden_channels,
            E=E,
            kernel_size=3,
            stride=1,
            padding=1,
            temperature=temperature,
            ratio=ratio,
        )
        self.conv2 = nn.Conv2d(hidden_channels, 4, kernel_size=3, stride=1, padding=1)
        self.pixelShuffle = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.pixelUnShuffle(x)
        out2 = self.conv1(out1)
        out3 = self.assemble2(self.assemble(out2))
        out4 = self.conv2(out3)
        result = self.pixelShuffle(out4)

        if self.global_residual:
            result = torch.add(result, x)

        return result
