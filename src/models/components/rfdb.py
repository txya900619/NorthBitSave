from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.components.conv import conv_layer


class ESA(nn.Module):
    """Enhanced Spatial Attention out_channels = n_feats."""

    def __init__(self, n_feats: int, conv: Type[nn.Module] = nn.Conv2d):
        super().__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RFDB(nn.Module):
    """Residual Feature Distillation Block out_channels = in_channels."""

    def __init__(self, in_channels: int):
        super().__init__()

        self.distilled_channels = in_channels // 2
        self.remaining_channels = in_channels
        self.conv1_distilled = conv_layer(in_channels, self.distilled_channels, kernel_size=1)
        self.conv1_remaining = conv_layer(in_channels, self.remaining_channels, kernel_size=3)
        self.conv2_distilled = conv_layer(
            self.remaining_channels, self.distilled_channels, kernel_size=1
        )
        self.conv2_remaining = conv_layer(
            self.remaining_channels, self.remaining_channels, kernel_size=3
        )
        self.conv3_distilled = conv_layer(
            self.remaining_channels, self.distilled_channels, kernel_size=1
        )
        self.conv3_remaining = conv_layer(
            self.remaining_channels, self.remaining_channels, kernel_size=3
        )
        self.conv4 = conv_layer(self.remaining_channels, self.distilled_channels, kernel_size=3)
        self.activation = nn.LeakyReLU()
        self.conv5 = conv_layer(self.distilled_channels * 4, in_channels, kernel_size=1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input: Tensor) -> Tensor:
        distilled_out1 = self.activation(self.conv1_distilled(input))
        remain_out1 = self.conv1_remaining(input)
        remain_out1 = self.activation(remain_out1 + input)

        distilled_out2 = self.activation(self.conv2_distilled(remain_out1))
        remain_out2 = self.conv2_remaining(remain_out1)
        remain_out2 = self.activation(remain_out2 + remain_out1)

        distilled_out3 = self.activation(self.conv3_distilled(remain_out2))
        remain_out3 = self.conv3_remaining(remain_out2)
        remain_out3 = self.activation(remain_out3 + remain_out2)

        remain_out4 = self.activation(self.conv4(remain_out3))

        out = torch.cat([distilled_out1, distilled_out2, distilled_out3, remain_out4], dim=1)
        out_fused = self.esa(self.conv5(out))

        return out_fused


if __name__ == "__main__":
    _ = RFDB()
