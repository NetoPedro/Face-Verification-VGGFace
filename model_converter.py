from torch import nn
import torch
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

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

def convert_model(path="model.mdl"):
    new_model = VGGFace_Extractor()

    pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model_dict = new_model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)

    new_model.load_state_dict(pretrained_dict)

    new_model.save_weights("face_extractor_model.mdl")

convert_model()