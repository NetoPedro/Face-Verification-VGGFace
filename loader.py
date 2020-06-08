from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import time
import os
import numpy as np
import torch
import matplotlib.pylab as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class FacesDataset(Dataset):
    """Faces Dataset"""

    def __init__(self, csv_file, root_dir, identities = 10, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        pd.DataFrame.from_records(data, columns =['Team', 'Age', 'Score'])    root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        x = pd.read_csv(csv_file)
        identities = np.linspace(1,identities,identities).astype(int)
        identities_existent = pd.DataFrame(x.iloc[:,0].str.split("/").tolist())
        identities_existent = identities_existent.iloc[:,0].str.replace("n","").astype(int)
        self.x = x[identities_existent.isin(identities)]
        self.size = len(self.x)
        print(self.size)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.x.iloc[idx, 0])
        image = plt.imread(img_name)
        identity = int(img_name.split("/n0")[1].split("/")[0])

        if self.transform:
            image = self.transform(image)
       
        sample = {'image': image, 'identity': (int(identity) - 2)}
        return sample


class TripletDataset(Dataset):

    def __init__(self,csv_file,root_dir,identities = 10,transform = None):

        x = pd.read_csv(csv_file)
        identities = np.linspace(1,identities,identities).astype(int)
        identities_existent = pd.DataFrame(x.iloc[:,0].str.split("/").tolist())
        identities_existent = identities_existent.iloc[:,0].str.replace("n","").astype(int)
        self.x = x[identities_existent.isin(identities)]
        self.x = self.generate_triplets(x,int(len(self.x)*0.18))
        self.size = len(self.x)
        print(self.size)
        self.root_dir = root_dir
        self.transform = transform

    def generate_triplets(self,x,n = 10):
        print("Generating: " + str(n) + " triplets")
        triplets = [("","","") for k in range(n)]
        size = len(x)
        identities = pd.DataFrame(x.iloc[:,0].str.split("/").tolist()).iloc[:,0]
        
        
        for i in range(n): 
            if i % 3000 == 0:
                print("Progress: "+ str(i/n * 100) +"%\tTriplet n: " + str(i)+"/"+str(n))
            anchor = x.iloc[np.random.randint(size),0]
            
            is_positive = identities.isin([anchor.split("/")[0]])
            
            positive_images = x[is_positive]
            positive = ""
            positive_size = len(positive_images)
            
                        
            while True:
                positive = positive_images.iloc[np.random.randint(positive_size),0]
               
                if positive != anchor:
                    break
                
            
            negative_images = x[~is_positive]
            negative_size = len(negative_images)
            negative = negative_images.iloc[np.random.randint(negative_size),0]
           
            triplets[i] = (anchor,positive,negative)
        

        return pd.DataFrame.from_records(triplets, columns =['Anchor', 'Positive', 'Negative']) 

    def __len__(self):
        return self.size

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        anchor,positive,negative = self.x.iloc[idx]
        
        anchor_img_name = os.path.join(self.root_dir,anchor)
        positive_img_name = os.path.join(self.root_dir,positive)
        negative_img_name = os.path.join(self.root_dir,negative)

        anchor_image = plt.imread(anchor_img_name)
        positive_image = plt.imread(positive_img_name)
        negative_image = plt.imread(negative_img_name)

       
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        sample = {'anchor': anchor_image, 'positive': positive_image, 'negative': negative_image}
        return sample

        

        

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
