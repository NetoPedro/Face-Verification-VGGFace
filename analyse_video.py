import numpy as np
import model as Model
import torch
import utils
from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import crop_resize,get_size
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import glob
import re
import ffmpeg
from argparse import ArgumentParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def check_rotation(path_video_file):
    """Source: https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting?fbclid=IwAR3OcgKRaJoPi9ljJ6UAlnaGKqbnzk6H5uaAMDmD8_ZHE86xa-7gbVlPGD0"""
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode

def correct_rotation(frame, rotateCode):
    "Source: https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting?fbclid=IwAR3OcgKRaJoPi9ljJ6UAlnaGKqbnzk6H5uaAMDmD8_ZHE86xa-7gbVlPGD0"
    return cv2.rotate(frame, rotateCode)


def process_video(weights_path,video_path,output_path,margins=40,facenet_threshold=.985,euclidean_distance_threshold = 120.0):
    """
    Reads an input video, and outputs another similar video with the faces identified. Ignores Unknowns.
    :param weights_path: The path to the weights of the model
    :param video_path: The path to the video to be processed
    :param output_path: The path to output the video to be stored
    :param margins: Margin to add to the size of the bounding box
    :param facenet_threshold: Threshold of confidence while detecting faces
    :param euclidean_distance_threshold: Threshold of the euclidean distance to identify the face
    """
    with torch.no_grad():
        mtcnn = MTCNN(image_size= 256, margin = 0)
        model = Model.VGGFace_Extractor().to(device)
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        cap = cv2.VideoCapture(video_path)
        rotateCode = check_rotation(video_path)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        ret, frame1 = cap.read()
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        i = 0
        while (cap.isOpened()):
            i += 1
            ret, frame2 = cap.read()
            if not (ret): break
            if rotateCode is not None:
                frame2  = correct_rotation(frame2, rotateCode)

            boxes, probs = mtcnn.detect(frame2)
            img_draw = frame2.copy()
            img_draw = Image.fromarray(img_draw)
            draw = ImageDraw.Draw(img_draw)
            if boxes is not None:
                names = []
                distances_difference = []
                for (box, point) in zip(boxes, probs):
                    """ Loop from the extract_face method from facenet_pytorch"""

                    if point < facenet_threshold: continue
                    margin = margins
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
                    best_distance = euclidean_distance_threshold + 5
                    for k,f1 in enumerate(files):
                        features = torch.load(f1)
                        distance = utils.euclidean_distance(features,features_1)
                        if distance < euclidean_distance_threshold and distance < best_distance:
                            best_distance = distance
                            name = re.sub('_[1-9]*[.]*[a-zA-Z]*', '', f1.replace(images_path,""))

                    names.append(name)
                    distances_difference.append(best_distance)

                for (box, point,name,distances) in zip(boxes, probs,names,distances_difference):
                    if point < facenet_threshold or name == "Unknown": continue
                    draw.rectangle(box.tolist(), width=4)
                    draw.text(box.tolist(), name, font=ImageFont.truetype("Keyboard.ttf",40))

            k = cv2.waitKey(3) & 0xff
            if k == 27:
                break
            out.write(np.asarray(img_draw))

        out.release()
        cap.release()
        cv2.destroyAllWindows()

def extract_features_individuals(weights_path):
    """
    Extracts the features from a group of faces
    :param weights_path: The path to the weights of the model
    """
    with torch.no_grad():
        model = Model.VGGFace_Extractor().to(device)
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        images_path = "individuals/"
        data_path = os.path.join(images_path, '*g')
        files = glob.glob(data_path)
        for k, f1 in enumerate(files):
            img = Image.open(f1).convert("RGB")
            features = model(utils.preprocess(img, device).reshape(-1, 3, 224, 224))
            torch.save(features,"individuals_extracted/"+re.sub('[.][a-zA-Z]*', '.pt',f1.split("/")[1]))


def main():
    argparse = ArgumentParser()

    argparse.add_argument('--weight_file_path', type=str, help='Pretrained weight file.', default="models/face_extractor_model.mdl")
    argparse.add_argument('--input_video_path', type=str, help='Input video path.',
                          default="./input_videos/video1.mp4")
    argparse.add_argument('--output_video_path', type=str, help='Path with the location to store the output video',
                          default="./output_videos/")
    argparse.add_argument('--margin', type=int, help='Margin to add to the size of the bounding box', default=120)
    argparse.add_argument('--extract_features', type=bool, help='Extract features from individuals dir', default=False)
    argparse.add_argument('--facenet_threshold', type=float, help='Threshold of confidence while detecting faces', default=.985)
    argparse.add_argument('--euclidean_distance_threshold', type=float, help='Threshold of the euclidean distance to identify the face', default=120.0)

    args = argparse.parse_args()


    # check arguments
    assert os.path.exists(args.weight_file_path), "Weights path : " + args.weight_file_path + " not found."
    assert os.path.exists(args.input_video_path), "video path: " + args.input_video_path + " not found."
    assert os.path.exists(args.output_video_path), "output directory: " + args.output_video_path + " not found."
    assert args.margin >= 0, "line_width should be >= 0."

    if args.extract_features:
        extract_features_individuals(args.weight_file_path)
    process_video(args.weight_file_path,
                  args.input_video_path,
                  args.output_video_path + "output_video.avi",
                  args.margin,
                  args.facenet_threshold,
                  args.euclidean_distance_threshold)

if __name__ == '__main__':
  main()
