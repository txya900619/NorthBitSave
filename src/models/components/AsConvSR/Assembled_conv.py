import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F


class ControlModule(nn.Module):
    """
    Control Module
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        temperature (int): Temperature of softmax
        ratio (int): Ratio of hidden channels to input channels
    """

    def __init__(self, in_channels, out_channels, temperature=30, ratio=4, E=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.E = E
        self.temperature = temperature

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        assert in_channels > ratio
        hidden_channels = in_channels // ratio

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, E * out_channels, kernel_size=1, bias=False),
        )

        self._initialize_weights()

    def forward(self, x):
        coeff = self.avgpool(x)  # bs, channels, 1, 1
        coeff = self.net(coeff)
        coeff = rearrange(coeff, "b (oc e) 1 1 -> b oc e", e=self.E)  # bs, channels
        # coeff = self.net(coeff).reshape(x.shape[0], -1)  # bs, E * out_channels
        # coeff = coeff.reshape(coeff.shape[0], self.out_channels, self.E)
        coeff = F.softmax(coeff / self.temperature, dim=2)
        return coeff

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AssembledBlock(nn.Module):
    """Assembled Block.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        padding (int): Zero-padding added to both sides of the input
        dilation (int): Spacing between kernel elements
        groups (int): Number of blocked connections from input channels to output channels
        bias (bool): If ``True``, adds a learnable bias to the output
        E (int): Number of experts
        temperature (int): Temperature of softmax
        ratio (int): Ratio of hidden channels to input channels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        E=4,
        temperature=30,
        ratio=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.E = E
        self.temperature = temperature
        self.ratio = ratio

        self.control_module = ControlModule(in_channels, out_channels, temperature, ratio, E)
        self.weight1 = nn.Parameter(
            torch.randn(E, out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=True,
        )
        self.weight2 = nn.Parameter(
            torch.randn(E, out_channels, out_channels // groups, kernel_size, kernel_size),
            requires_grad=True,
        )
        self.weight3 = nn.Parameter(
            torch.randn(E, out_channels, out_channels // groups, kernel_size, kernel_size),
            requires_grad=True,
        )

        if self.bias:
            self.bias1 = nn.Parameter(
                torch.randn(E, out_channels), requires_grad=True
            )  # E, out_channels
            self.bias2 = nn.Parameter(
                torch.randn(E, out_channels), requires_grad=True
            )  # E, out_channels
            self.bias3 = nn.Parameter(
                torch.randn(E, out_channels), requires_grad=True
            )  # E, out_channels

    def forward(self, x: Tensor):
        bs, in_channels, h, w = x.shape
        coeff = self.control_module(x)  # bs, out_channels, E
        weight1 = rearrange(self.weight1, "e oc ic k1 k2 -> e oc (ic k1 k2)")
        weight2 = rearrange(self.weight2, "e oc ic k1 k2 -> e oc (ic k1 k2)")
        weight3 = rearrange(self.weight3, "e oc ic k1 k2 -> e oc (ic k1 k2)")

        x = rearrange(x, "b ic h w -> () (b ic) h w")  # 1, bs * in_channels, h, w

        aggregate_weight1 = torch.zeros(
            bs,
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        ).to(
            x.device
        )  # bs, out_channels, in_channels // groups, k, k
        aggregate_weight2 = torch.zeros(
            bs,
            self.out_channels,
            self.out_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        ).to(
            x.device
        )  # bs, out_channels, in_channels // groups, k, k
        aggregate_weight3 = torch.zeros(
            bs,
            self.out_channels,
            self.out_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        ).to(
            x.device
        )  # bs, out_channels, in_channels // groups, k, k
        if self.bias:
            aggregate_bias1 = torch.zeros(bs, self.out_channels).to(x.device)  # bs, out_channels
            aggregate_bias2 = torch.zeros(bs, self.out_channels).to(x.device)  # bs, out_channels
            aggregate_bias3 = torch.zeros(bs, self.out_channels).to(x.device)  # bs, out_channels

        # use einsum instead of for loop to speed up backpropagation
        aggregate_weight1 = torch.einsum("bce,ech->bch", coeff, weight1)
        aggregate_weight1 = rearrange(
            aggregate_weight1,
            "b oc (num_groups k1 k2) -> (b oc) num_groups k1 k2",
            k1=self.kernel_size,
            k2=self.kernel_size,
        )

        aggregate_weight2 = torch.einsum("bce,ech->bch", coeff, weight2)
        aggregate_weight2 = rearrange(
            aggregate_weight2,
            "b oc (num_groups k1 k2) -> (b oc) num_groups k1 k2",
            k1=self.kernel_size,
            k2=self.kernel_size,
        )

        aggregate_weight3 = torch.einsum("bce,ech->bch", coeff, weight3)
        aggregate_weight3 = rearrange(
            aggregate_weight3,
            "b oc (num_groups k1 k2) -> (b oc) num_groups k1 k2",
            num_groups=self.out_channels // self.groups,
            k1=self.kernel_size,
            k2=self.kernel_size,
        )

        if self.bias:
            aggregate_bias1 = torch.einsum("bce,ec->bc", coeff, self.bias1)
            aggregate_bias2 = torch.einsum("bce,ec->bc", coeff, self.bias2)
            aggregate_bias3 = torch.einsum("bce,ec->bc", coeff, self.bias3)

            aggregate_bias1 = rearrange(aggregate_bias1, "b oc -> (b oc)")
            aggregate_bias2 = rearrange(aggregate_bias2, "b oc -> (b oc)")
            aggregate_bias3 = rearrange(aggregate_bias3, "b oc -> (b oc)")
        else:
            aggregate_bias1, aggregate_bias2, aggregate_bias3 = None, None, None

        out = F.conv2d(
            x,
            weight=aggregate_weight1,
            bias=aggregate_bias1,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * bs,
        )  # bs * out_channels, in_channels // groups, h, w
        out = F.conv2d(
            out,
            weight=aggregate_weight2,
            bias=aggregate_bias2,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * bs,
        )  # bs * out_channels, in_channels // groups, h, w
        out = F.conv2d(
            out,
            weight=aggregate_weight3,
            bias=aggregate_bias3,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * bs,
        )  # bs * out_channels, in_channels // groups, h, w
        out = out.reshape(bs, self.out_channels, out.shape[2], out.shape[3])

        return out
