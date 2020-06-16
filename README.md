# Face Recognition

This project aims to identify people in videos with deep learning methods. 
The initial idea was to replicate the [VGG2 Dataset paper](https://arxiv.org/abs/1710.08092), therefore I have trained a ResNet-50 model for that purpose on a subset of that dataset. 
Since the focus was in the identification of a person by their face, the component regarding the detection of faces is a pretrained [MTCNN Model](https://arxiv.org/abs/1604.02878) using the weights given in this [facenet-pytorch repository](https://github.com/timesler/facenet-pytorch).

One of the important elements of the proposed solution is that the algorithm does not require to be trained with the information of the people that we are aiming to recognize. The trained model output is given in a embedding space with size 1024. With the information of this output and at least 1 picture of the person that we aim to identify, it is possible to calculate the euclidean distance between the embedding of the target and the embedding of the face captured in the video. If the distance is below a specific threshold, then it corresponds to the person that we want to find. 

## Demo 

## Instructions to use
 




