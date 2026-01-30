import sys
import os

# Get the path to the project root (3 levels up from this script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch.nn as nn
import torch.nn.functional as F
import torch
from src.models.domain_adaptation.utils import ReverseLayerF
from torch.nn.utils import spectral_norm

class Extractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4,  # inclus maintenant ici
        )

    def forward(self, x):
        return self.features(x)

# Sous-modèle 2 : AdaptiveAvgPool2d et la couche finale Linear
class Classifier(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.pool = original_model.avgpool
        self.classifier = original_model.fc

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.linear1 = spectral_norm(nn.Linear(in_features, out_features))
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = spectral_norm(nn.Linear(out_features, out_features))
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = (
            nn.Sequential() if in_features == out_features
            else spectral_norm(nn.Linear(in_features, out_features))
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.linear1(x)), 0.2)
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out += identity
        out = F.leaky_relu(out, 0.2)
        return out

# 6 layers
# class Discriminator(nn.Module):
#     def __init__(self, encoder_output_size):
#         super(Discriminator, self).__init__()
#         self.flatten = nn.Flatten()
#         self.block1 = ResidualBlock(encoder_output_size, 1024)
#         self.block2 = ResidualBlock(1024, 512)
#         self.block3 = ResidualBlock(512, 256)
#         self.out = spectral_norm(nn.Linear(256, 2))  # Binary classification
#
#     def forward(self, input_feature, alpha):
#         x = ReverseLayerF.apply(input_feature, alpha)
#         x = self.flatten(x)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         return self.out(x)

# class Discriminator(nn.Module):
#     def __init__(self, encoder_output_size):
#         super(Discriminator, self).__init__()
#         self.discriminator = nn.Sequential(
#             nn.Linear(in_features=encoder_output_size, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=2)
#         )
#
#     def forward(self, input_feature, alpha):
#         # Reverse the gradients (if you're using a gradient reversal layer)
#         reversed_input = ReverseLayerF.apply(input_feature, alpha)
#
#         # Flatten the input feature from (batch_size, 512, height, width) to (batch_size, flattened_size)
#         flattened_input = reversed_input.view(reversed_input.size(0), -1)  # Flatten the feature map
#
#         # Pass through the discriminator's layers
#         x = self.discriminator(flattened_input)
#         return x

class Discriminator(nn.Module):
    def __init__(self, encoder_output_size):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=encoder_output_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2)
        )

    def forward(self, input_feature, alpha):
        # Reverse the gradients (if you're using a gradient reversal layer)
        reversed_input = ReverseLayerF.apply(input_feature, alpha)

        # Flatten the input feature from (batch_size, 512, height, width) to (batch_size, flattened_size)
        flattened_input = reversed_input.view(reversed_input.size(0), -1)  # Flatten the feature map

        # Pass through the discriminator's layers
        x = self.discriminator(flattened_input)
        return x

class DANNModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(DANNModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        x = self.encoder(x)  # Passer par l'encodeur
        x = self.classifier(x)  # Passer par le classificateur
        return x