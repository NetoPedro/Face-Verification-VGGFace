import torch
import os
import numpy as np
import random
from ast import literal_eval
import csv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def load_image(image):
    '''Loads an image from the test set.

                Args:
                    image (str): The relative path of the image in the test case.


                Returns:
                    image_blob: The image loaded as an Numpy matrix.
                '''
    img = plt.imread(image)
    return img

#def extract_individuals_features():




class SubtractMean(object):
            """Convert ndarrays in sample to Tensors."""
            mean_bgr = np.array([91.4953, 103.8827, 131.0912])
            def __call__(self, img):
                img = np.array(img, dtype=np.uint8)
                img = img[:, :, ::-1]  # RGB -> BGR
                img = img.astype(np.float32)
                img -= self.mean_bgr
                img = img.transpose(2, 0, 1)  # C x H x W
                img = torch.from_numpy(img).float()
                return img

def preprocess(image,device):
    transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),SubtractMean()])
    image = transform(image)
   
    return image.to(device)


#create_test_cases("/l/Dataset/test/", "/l/Dataset/test_cases.csv", 10, 4000)
#load_test_cases( "../Dataset/test_cases.csv")

def euclidean_distance(feature1, feature2):
        """
            Args:
                original_embedding (tensor): Tensor containing the embedding space of the original image. Arbitrary size.
                outputs (tensor): Tensor that contais the embeddings of all other images.
                images_count (int): Number of different images on outputs
            Returns:
                distance (tensor): Tensor of size "images_count" with euclidean distance for each image between the original
        """
        return torch.cdist(feature1.view(1, -1), feature2.view(1, -1))