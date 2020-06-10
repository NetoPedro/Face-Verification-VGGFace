from torch import nn
import torch
from torchvision import models
from torchvision.models import resnet50
import  numpy as np


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VGGFace_training(nn.Module):
    def __init__(self,identities=1000):
        super(VGGFace_training,self).__init__()
        self.model = models.resnet50(True)
        self.model.fc = Identity()
        self.model.avgpool = Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048,identities)
        self.embed_layer = nn.Linear(2048,1024)

    def forward(self, x,triplet = False):
        x = self.model(x).reshape(x.shape[0],-1,7,7)
        out = self.avgpool(x)
        if not(triplet):
            out = self.fc(out.flatten(start_dim=1))
        else:
            out = out.reshape(out.shape[0], -1)
            x = out / torch.norm(out, 2,dim=1).reshape(out.shape[0],1)
            embedding = self.embed_layer(x)
            return out,embedding
        return  out, x

    def triplet_mode(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.embed_layer.requires_grad = True

    def save_weights(self,path):
        torch.save(self.state_dict(), path)



class VGGFace_Extractor(nn.Module):
    def __init__(self):
        super(VGGFace_Extractor,self).__init__()
        self.model = models.resnet50(True)
        self.model.fc = Identity()
        self.model.avgpool = Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_layer = nn.Linear(2048,1024)

    def forward(self, x):
        x = self.model(x).reshape(x.shape[0],-1,7,7)
        out = self.avgpool(x)

        out = out.reshape(out.shape[0], -1)
        x = out / torch.norm(out, 2,dim=1).reshape(out.shape[0],1)
        embedding = self.embed_layer(x)
        return out,embedding


    def triplet_mode(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.embed_layer.requires_grad = True

    def save_weights(self,path):
        torch.save(self.state_dict(), path)

