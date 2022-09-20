import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class TSCCEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode=0):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.linear_projector = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvModule(hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3)
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.linear_projector(x)

        if self.training:
            mask = self.mask_mode
        else:
            mask = 1

        if mask == 0:
            mask = torch.from_numpy(np.random.binomial(1, 0.5, size=(x.size(0), x.size(1)).to(x.device))).to(torch.bool)
        elif mask == 1:
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)

        mask &= nan_mask
        x[~mask] = 0

        x = x.transpose(1, 2)
        x = self.repr_dropout(self.feature_extractor(x))
        x = x.transpose(1, 2)

        return x


class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.DilatedConv1 = DilatedConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.DilatedConv2 = DilatedConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.ResProjector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.ResProjector is None else self.ResProjector(x)
        x = F.gelu(x)
        x = self.DilatedConv1(x)
        x = F.gelu(x)
        x = self.DilatedConv2(x)
        return x + residual


class DilatedConvModule(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            DilatedConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)

