import numpy as np
import model as Model
import torch
import utils
from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import crop_resize,get_size
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import os
import glob
import re


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def process_video(video_path):
    with torch.no_grad():
        mtcnn = MTCNN(image_size= 256, margin = 0)
        model = Model.VGGFace_Extractor().to(device)
        model.load_state_dict(torch.load("models/face_extractor_model.mdl"))
        model.eval()
        threshold = 115.0
        #with open('threshold.txt') as f:
        #    threshold = float(f.readline())
        #pretrained_dict = torch.load("models/face_extractor_model.mdl", map_location=lambda storage, loc: storage)
        #model.load_state_dict(pretrained_dict)

        cap = cv2.VideoCapture(video_path)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        #out = cv2.VideoWriter('output_videos/output2.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 400)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 5

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        i = 0

        while (cap.isOpened()):
            i += 1

            ret, frame2 = cap.read()
            if not (ret): break

            boxes, probs = mtcnn.detect(frame2)

            img_draw = frame2.copy()
            img_draw = Image.fromarray(img_draw)
            draw = ImageDraw.Draw(img_draw)
            face = None

            if boxes is not None:
                names = []
                distances_difference = []
                for (box, point) in zip(boxes, probs):
                    """ Loop inspired by the extract face method from facenet_pytorch"""
                    if point < .60: continue
                    margin = 40
                    image_size = 256
                    margin = [
                        margin * (box[2] - box[0]) / (image_size - margin),
                        margin * (box[3] - box[1]) / (image_size - margin),
                    ]
                    raw_image_size = get_size(img_draw)
                    box = [
                        int(max(box[0] - margin[0] / 2, 0)),
                        int(max(box[1] - margin[1] / 2, 0)),
                        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
                        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
                    ]

                    face = img_draw.crop(box).copy().resize((image_size, image_size), Image.BILINEAR).convert("RGB")
                    features_1 = model(utils.preprocess(face,device).reshape(-1, 3, 224, 224))

                    images_path = "individuals_extracted/"
                    data_path = os.path.join(images_path, '*pt')
                    files = glob.glob(data_path)
                    name = "Unknown"
                    best_distance = threshold + 5
                    for k,f1 in enumerate(files):
                        #img = Image.open(f1).convert("RGB")
                        features = torch.load(f1)#model(utils.preprocess(img,device).reshape(-1, 3, 224, 224))
                        distance = utils.euclidean_distance(features,features_1)
                        if distance < threshold and distance < best_distance:
                            best_distance = distance
                            name = re.sub('_[1-9]*[.]*[a-zA-Z]*', '', f1.replace(images_path,""))




                    names.append(name)
                    distances_difference.append(best_distance)

                for (box, point,name,distances) in zip(boxes, probs,names,distances_difference):
                    if point < .60: continue
                    draw.rectangle(box.tolist(), width=4)
                    print(name + "  " + str(distances))
                    draw.text(box.tolist(), name, font=ImageFont.truetype("Keyboard.ttf",40))

                #plt.imshow(img_draw)
                #plt.show()
            cv2.imshow("",np.asarray(img_draw))

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
         #   out.write(np.asarray(img_draw))

        print("Video Generated")
        #out.release()
        cap.release()
        cv2.destroyAllWindows()

def extract_features_individuals():
    with torch.no_grad():
        model = Model.VGGFace_Extractor().to(device)
        model.load_state_dict(torch.load("models/face_extractor_model.mdl"))
        model.eval()
        images_path = "individuals/"
        data_path = os.path.join(images_path, '*g')
        files = glob.glob(data_path)
        for k, f1 in enumerate(files):
            img = Image.open(f1).convert("RGB")
            features = model(utils.preprocess(img, device).reshape(-1, 3, 224, 224))
            #features = features.numpy()
            torch.save(features,"individuals_extracted/"+re.sub('[.][a-zA-Z]*', '.pt',f1.split("/")[1]))

#extract_features_individuals()
process_video("input_videos/video2.mp4")