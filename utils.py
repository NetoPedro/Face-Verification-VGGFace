import torch
import os
import numpy as np
import random
from ast import literal_eval
import csv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean_bgr = np.array([91.4953, 103.8827, 131.0912])
def create_test_cases(test_set_path, destination_path, n, k):
    '''Creates test cases corresponding to the few shots specifications.

        Args:
            test_set_path (str): The path to the test set.
            destination_path (str): The path to create the new file.
            n (int): Number of images per case.
            k (int): Number of test cases.

        Returns:
            success: If the process was successful or not.
        '''
    test_cases = []

    identities = next(os.walk(test_set_path))[1] # All the persons in the dataset
    identities_count = len(identities)  # The number of persons in the dataset

    for i in range(0,k,1): # For each test case to create

        anchor_identity_index = np.random.randint(0, identities_count)  # Selects the anchor
        anchor_identity = identities[anchor_identity_index]             # Selects the identity of the anchor
        path_to_anchor_folder = test_set_path + str(anchor_identity) + "/"  # Path to the anchor folder
        images_from_anchor_identity = next(os.walk(path_to_anchor_folder))[2]  # Loads names of images in anchor folder
        anchor_image = ""  # Anchor image to be defined below

        number_of_true_images = np.random.randint(1, n)  # Number of images from the anchor from the n samples

        other_identities_indices = np.random.randint(0, identities_count, size=(n-number_of_true_images))

        images = [] # n sample images for this test case

        for index in other_identities_indices: # For each identity_index
            while index == anchor_identity_index: # Changes the index if it is the same as the anchor
                index = np.random.randint(0,identities_count)

            path_to_identity_folder = test_set_path + str(identities[index]) + "/"  # Path to this identity folder
            images_from_identity = next(os.walk(path_to_identity_folder))[2]  # Loads names of images from folder
            images.append((path_to_identity_folder +
                           images_from_identity[np.random.randint(0,len(images_from_identity))],0)) # Randomly add image

        removable = False # If there must be no repeated images in the sampled images
        if number_of_true_images < len(images_from_anchor_identity):  # Only allows repetitions in this case
            removable = True

        for j in range(0,number_of_true_images+1):  # For each sample image from the anchor + the anchor
            if j == 0: # If it is the anchor
                index = np.random.randint(0,len(images_from_anchor_identity))  # Randomly selects an image
                anchor_image = path_to_anchor_folder + images_from_anchor_identity[index]  # Defines the anchor
                images_from_anchor_identity.remove(images_from_anchor_identity[index])  # Removes the image
            else:
                index = np.random.randint(0, len(images_from_anchor_identity))  # Randomly selects an image
                images.append((path_to_anchor_folder + images_from_anchor_identity[index],1))  # Adds image to samples
                if removable: # If repetitions are not allowed
                    images_from_anchor_identity.remove(images_from_anchor_identity[index])  # Removes the image

        random.shuffle(images)  # Shuffles images so positive and negative cases become mixed
        test_cases.append((anchor_image,images))  # Appends a tuple with the anchor and the samples to the test_cases

    with open(destination_path, 'w+', newline='') as myfile:  # Saves the each case as a line of a csv file.
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for case in test_cases:
            wr.writerow(case)

    return True


def load_test_cases(path):
    '''Loads the test cases.

            Args:
                path (str): The path to the test cases.


            Returns:
                test_cases: list of tuples, first element is the original image, the second is a list of other images.
            '''
    test_cases = []
    with open(path, 'r', newline='') as myfile: # Loads the test_cases from the csv file
        reader = csv.reader(myfile)
        for case in reader:
            caseToList = (case[0],literal_eval(case[1])) # Saves as a tuple with string and list (converted from string)
            test_cases.append(caseToList)  # Appends to test_cases

    return test_cases



def load_image(image):
    '''Loads an image from the test set.

                Args:
                    image (str): The relative path of the image in the test case.


                Returns:
                    image_blob: The image loaded as an Numpy matrix.
                '''

   
    img = plt.imread(image)
    return img




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
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256),transforms.CenterCrop(224),SubtractMean()])
    image = transform(image)
   
    return image.to(device)


#create_test_cases("/l/Dataset/test/", "/l/Dataset/test_cases.csv", 10, 4000)
#load_test_cases( "../Dataset/test_cases.csv")
