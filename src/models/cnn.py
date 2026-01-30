import torch.nn as nn
import torch.nn.functional as F

from src.models.conf_models import Activationfunction


# From https://github.com/georgeretsi/HTR-best-practices/
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, act_fct=nn.ReLU()):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.act_fct = act_fct

    def forward(self, x):
        out = self.act_fct(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act_fct(out)

        return out


# https://medium.com/@tahasamavati/squeeze-and-excitation-explained-387b5981f249
class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class CNN(nn.Module):
    def __init__(self, cnn_cfg,
                 activation_fct=Activationfunction.RELU,
                 load_img_as_grayscale=1,
                 add_squeeze_excitation=1):
        super(CNN, self).__init__()

        self.k = 1

        act_fct = nn.ReLU()

        if activation_fct == Activationfunction.LEAKY_RELU:
            act_fct = nn.LeakyReLU()
        elif activation_fct == Activationfunction.SILU:
            act_fct = nn.SiLU()

        dim_in = 1
        if load_img_as_grayscale != 1:
            dim_in = 3

        self.features = nn.ModuleList([nn.Conv2d(dim_in, 32, 7, [2, 2], 3), act_fct])

        in_channels = 32
        cntm = 0
        cnt = 1
        cnt_se = 0
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            elif m == 'MH':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x, act_fct=act_fct))

                    in_channels = x
                    cnt += 1

                    if add_squeeze_excitation == 1:
                        self.features.add_module('se' + str(cnt_se), SE_Block(in_channels, x))

                        cnt_se += 1

    def forward(self, x):
        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

        y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k // 2])

        return y
