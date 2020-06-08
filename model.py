from torch import nn
import torch
from torchvision import models

class TinyFaces(nn.Module):
    pass


class FaceFeatureExtractor(nn.Module):
    pass


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

