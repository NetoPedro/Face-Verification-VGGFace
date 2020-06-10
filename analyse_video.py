import numpy as np
import model as Model
import torch
import utils
from facenet_pytorch import MTCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def process_video(video_path):
    mtcnn = MTCNN(image_size= 224, margin = 10)
    model = Model.VGGFace_Extractor().to(device)

