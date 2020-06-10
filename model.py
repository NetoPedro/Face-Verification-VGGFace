from torch import nn
import torch
from torchvision import models
from torchvision.models import resnet50
import  numpy as np


class FaceFeatureExtractor(nn.Module):
    pass


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


class DetectionModel(nn.Module):
    """
    Hybrid Model from Tiny Faces paper
    Source: https://github.com/varunagrawal/tiny-faces-pytorch/blob/master/models/model.py
    """

    def __init__(self, base_model=resnet50, num_templates=1, num_objects=1):
        super().__init__()
        # 4 is for the bounding box offsets
        output = (num_objects + 4)*num_templates
        self.model = base_model(pretrained=True)

        # delete unneeded layer
        del self.model.layer4

        self.score_res3 = nn.Conv2d(in_channels=512, out_channels=output,
                                    kernel_size=1, padding=0)
        self.score_res4 = nn.Conv2d(in_channels=1024, out_channels=output,
                                    kernel_size=1, padding=0)

        self.score4_upsample = nn.ConvTranspose2d(in_channels=output, out_channels=output,
                                                  kernel_size=4, stride=2, padding=1, bias=False)
        self._init_bilinear()

    def _init_weights(self):
        pass

    def _init_bilinear(self):
        """
        Initialize the ConvTranspose2d layer with a bilinear interpolation mapping
        :return:
        """
        k = self.score4_upsample.kernel_size[0]
        factor = np.floor((k+1)/2)
        if k % 2 == 1:
            center = factor
        else:
            center = factor + 0.5
        C = np.arange(1, 5)

        f = np.zeros((self.score4_upsample.in_channels,
                      self.score4_upsample.out_channels, k, k))

        for i in range(self.score4_upsample.out_channels):
            f[i, i, :, :] = (np.ones((1, k)) - (np.abs(C-center)/factor)).T @ \
                            (np.ones((1, k)) - (np.abs(C-center)/factor))

        self.score4_upsample.weight = torch.nn.Parameter(data=torch.Tensor(f))

    def learnable_parameters(self, lr):
        parameters = [
            # Be T'Challa. Don't freeze.
            {'params': self.model.parameters(), 'lr': lr},
            {'params': self.score_res3.parameters(), 'lr': 0.1*lr},
            {'params': self.score_res4.parameters(), 'lr': 1*lr},
            {'params': self.score4_upsample.parameters(), 'lr': 0}  # freeze UpConv layer
        ]
        return parameters

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        # res2 = x

        x = self.model.layer2(x)
        res3 = x

        x = self.model.layer3(x)
        res4 = x

        score_res3 = self.score_res3(res3)

        score_res4 = self.score_res4(res4)
        score4 = self.score4_upsample(score_res4)

        # We need to do some fancy cropping to accomodate the difference in image sizes in eval
        if not self.training:
            # from vl_feats DagNN Crop
            cropv = score4.size(2) - score_res3.size(2)
            cropu = score4.size(3) - score_res3.size(3)
            # if the crop is 0 (both the input sizes are the same)
            # we do some arithmetic to allow python to index correctly
            if cropv == 0:
                cropv = -score4.size(2)
            if cropu == 0:
                cropu = -score4.size(3)

            score4 = score4[:, :, 0:-cropv, 0:-cropu]
        else:
            # match the dimensions arbitrarily
            score4 = score4[:, :, 0:score_res3.size(2), 0:score_res3.size(3)]

        score = score_res3 + score4

        return score