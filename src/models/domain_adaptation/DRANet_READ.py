import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from .batchinstancenorm import BatchInstanceNorm2d as Normlayer
import functools
from functools import partial
import torchvision.transforms as ttransforms


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                bin(filters)
            )

    def forward(self, x):
        output = self.main(x)
        output += self.shortcut(x)
        return output


class Encoder(nn.Module):
    def __init__(self, channels=1):
        super(Encoder, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            bin(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Separator(nn.Module):
    def __init__(self, imsize, converts, ch=256, down_scale=2):
        super(Separator, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
        )
        self.w = nn.ParameterDict()
        w, h = imsize
        for cv in converts:
            self.w[cv] = nn.Parameter(torch.ones(1, ch, h//down_scale, w//down_scale), requires_grad=True)

    def forward(self, features, converts=None):
        contents, styles = dict(), dict()
        for key in features.keys():
            styles[key] = self.conv(features[key])
            contents[key] = features[key] - styles[key]
            if '2read_' in key:
                source, target = key.split('2read_', 1)
                target = 'read_' + target
                contents[target] = contents[key]

        if converts is not None:
            for cv in converts:
                source, target = cv.split("2read_", 1)
                target = "read_" + target
                print("source: " + source)
                print("target: " + target)
                w = F.interpolate(self.w[cv], size=contents[source].shape[2:], mode='bilinear', align_corners=False)
                contents[cv] = w * contents[source]
        return contents, styles


class Generator(nn.Module):
    def __init__(self, out_height, out_width_with_pad, input_channels=256):  # Match your encoder output channels
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # e.g., 6x6 -> 12x12
            spectral_norm(nn.ConvTranspose2d(input_channels, 256, 3, 1, 1)),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 12x12 -> 24x24
            spectral_norm(nn.ConvTranspose2d(256, 128, 3, 1, 1)),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 24x24 -> 48x48
            spectral_norm(nn.ConvTranspose2d(128, 64, 3, 1, 1)),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(64, 1, 3, 1, 1)),
            nn.Tanh()
        )
        self.out_height = out_height
        self.out_width_with_pad = out_width_with_pad

    def forward(self, content, style):
        x = content + style
        out = self.model(x)
        # out = F.interpolate(out, size=(160, 1826), mode='bilinear', align_corners=False)
        out = F.interpolate(out, size=(self.out_height, self.out_width_with_pad), mode='bilinear', align_corners=False)
        return out





class Classifier(nn.Module):
    def __init__(self, channels=1, num_classes=33):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(6912, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features

        # Modify first conv layer to accept 1 channel instead of 3
        old_conv = features[0]
        new_conv = nn.Conv2d(1, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding)

        # Initialize new_conv weights by averaging old weights across RGB channels
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.bias[:] = old_conv.bias

        features[0] = new_conv

        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_4_2 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2, 7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7, 12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12, 21):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(21, 25):
            self.to_relu_4_2.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_4_2(h)
        h_relu_4_2 = h
        out = (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_4_2)
        return out

class Discriminator_USPS(nn.Module):
    def __init__(self, channels=1):
        super(Discriminator_USPS, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(2304, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


class Discriminator_MNIST(nn.Module):
    def __init__(self, channels=1):
        super(Discriminator_MNIST, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))  # 🔧 add this line
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 1),  # 🔧 now input is always 256 (256×1×1)
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        return self.fc(output)



class PatchGAN_Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(PatchGAN_Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)

