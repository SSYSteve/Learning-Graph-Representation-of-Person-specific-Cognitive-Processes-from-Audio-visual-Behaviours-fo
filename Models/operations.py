import torch
import torch.nn as nn

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3': lambda C, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3': lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
}

OPS_DOWN = {
    'avg_pool_3': lambda C, stride, affine: nn.AvgPool1d(3, stride=2, padding=1, count_include_pad=False),
    'max_pool_3': lambda C, stride, affine: nn.MaxPool1d(3, stride=2, padding=1),
    'sep_conv_3': lambda C, stride, affine: SepConv(C, C, 3, 2, 1, affine=affine),
    'sep_conv_5': lambda C, stride, affine: SepConv(C, C, 5, 2, 2, affine=affine),
    'dil_conv_3': lambda C, stride, affine: DilConv(C, C, 3, 2, 2, 2, affine=affine),
    'dil_conv_5': lambda C, stride, affine: DilConv(C, C, 5, 2, 4, 2, affine=affine),
}

OPS_UP = {
    'tran_conv_3': lambda C, stride, affine: TranConv(C, C, 3, 2, 1, 1, affine),
    'linear': lambda C, stride, affine: nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
    'nearest': lambda C, stride, affine: nn.Upsample(scale_factor=2, mode='nearest'),
}


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        # assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv1d(C_in, C_out//2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv1d(C_in, C_out//2, 1, stride=2, padding=0, bias=False)
        self.conv_3 = nn.Conv1d((C_out//2)*2, C_out, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = self.conv_3(torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:])], dim=1))
        out = self.bn(out)
        return out


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class TranConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, output_padding, affine=True):
        super(TranConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose1d(C_in, C_out, kernel_size, stride, padding, output_padding),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv1d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class ReLUConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
        )

    def forward(self, x):
        return self.op(x)


class ReLUConvBNincrease(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBNincrease, self).__init__()
        # assert C_out % 2 == 0
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)
