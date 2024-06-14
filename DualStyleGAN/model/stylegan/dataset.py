from io import BytesIO
import os
import lmdb
from PIL import Image
from imageio import mimread
from skimage.transform import resize
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from functools import partial
from pathlib import Path
import torch
from skimage.color import gray2rgb
from torch import nn, einsum
import numpy as np
import cv2
import dlib
from skimage import io, img_as_float32
import glob
import torchvision
import random
import struct
EXTS = ['jpg', 'jpeg', 'png']

def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image

def exists(val):
    return val is not None

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)
def cv2line(img , x , y,crop_pts,line=True):
    for i in range(x,y):
        x1 = crop_pts[i * 2]
        y1 = crop_pts[i * 2 + 1]
        x2 = crop_pts[(i + 1) * 2]
        y2 = crop_pts[(i + 1) * 2 + 1]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
    if line:
        x_first = crop_pts[x * 2]
        y_first = crop_pts[x * 2 + 1]
        x_last = crop_pts[y * 2]
        y_last = crop_pts[y * 2 + 1]

        cv2.line(img, (int(x_last), int(y_last)), (int(x_first), int(y_first)), (255, 255, 255), 1)
    return img

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/featurize/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat')

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = mimread(name)
        if len(video[0].shape) == 2:
            video = [gray2rgb(frame) for frame in video]
        if frame_shape is not None:
            video = np.array([resize(frame, frame_shape) for frame in video])
        video = np.array(video)
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=True, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = frame_shape
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        if os.path.exists(os.path.join(root_dir, 'train')):#加载目录不要有train也需要有验证集
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)
        # print(train_videos)
        print(len(train_videos))
        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

    def read_imgs(self, path):
        frames = sorted(glob.glob(path + '/*.png'))
        num_frames = len(frames)
        if num_frames == 0:
            frames = sorted(glob.glob(path + '/*.jpg'))
            num_frames = len(frames)
        return frames, num_frames

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        is_pt = True
        while(is_pt):
            if self.is_train and self.id_sampling:
                name = self.videos[idx]

                path_all = glob.glob(os.path.join(self.root_dir, name + '#*.mp4'))
                temp_id = np.random.choice(path_all, replace=True, size=2)
                name_noid = np.random.choice(self.videos, replace=False, size=2)
                path_D = glob.glob(os.path.join(self.root_dir, name_noid[0] + '#*.mp4'))
                path_S = glob.glob(os.path.join(self.root_dir, name_noid[1] + '#*.mp4'))
                temp3 = np.random.choice(path_D, replace=True, size=1)
                temp4 = np.random.choice(path_S, replace=True, size=1)
                path = temp_id[0]
                path2 = temp_id[1]
                path3 = temp3[0]
                path4 = temp4[0]
                # print(path,path2,path3,path4)
            else:
                name = self.videos[idx]
                path = os.path.join(self.root_dir, name)

            if self.is_train and os.path.isdir(path):

                frames, num_frames = self.read_imgs(path)
                frame_idx = np.random.choice(num_frames)

                frames2, num_frames2 = self.read_imgs(path2)
                frame_idx2 = np.random.choice(num_frames2)

                frames3, num_frames3 = self.read_imgs(path3)
                frame_idx3 = np.random.choice(num_frames3)

                frames4, num_frames4 = self.read_imgs(path4)
                frame_idx4 = np.random.choice(num_frames4)

                frame_path = [os.path.join(path, frames[frame_idx]), os.path.join(path2, frames2[frame_idx2]),
                              os.path.join(path3, frames3[frame_idx3]), os.path.join(path4, frames4[frame_idx4])]
                img_D1 = Image.open(frame_path[0])
                img_S1 = Image.open(frame_path[1])
                img_D2 = Image.open(frame_path[2])
                img_S2 = Image.open(frame_path[3])
                gray = cv2.cvtColor(np.array(img_D1), cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                image1 = np.zeros((256, 256, 3), dtype=np.uint8)
                image1 = np.array(image1)
                crop_pts1 = []
                for face in faces:
                    # 预测面部关键点
                    landmarks = predictor(gray, face)
                    for n in range(0, 68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        crop_pts1.append(int(x))
                        crop_pts1.append(int(y))
                    image1 = cv2line(image1, 0, 16, crop_pts1, line=False)
                    image1 = cv2line(image1, 17, 21, crop_pts1)
                    image1 = cv2line(image1, 22, 26, crop_pts1)
                    image1 = cv2line(image1, 36, 41, crop_pts1)
                    image1 = cv2line(image1, 42, 47, crop_pts1)
                    image1 = cv2line(image1, 48, 59, crop_pts1)
                    image1 = cv2line(image1, 60, 67, crop_pts1)
                gray = cv2.cvtColor(np.array(img_D2), cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                image2 = np.zeros((256, 256, 3), dtype=np.uint8)
                image2 = np.array(image2)
                crop_pts2 = []
                for face in faces:
                    # 预测面部关键点
                    landmarks = predictor(gray, face)
                    for n in range(0, 68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        crop_pts2.append(int(x))
                        crop_pts2.append(int(y))
                    image2 = cv2line(image2, 0, 16, crop_pts2, line=False)
                    image2 = cv2line(image2, 17, 21, crop_pts2)
                    image2 = cv2line(image2, 22, 26, crop_pts2)
                    image2 = cv2line(image2, 36, 41, crop_pts2)
                    image2 = cv2line(image2, 42, 47, crop_pts2)
                    image2 = cv2line(image2, 48, 59, crop_pts2)
                    image2 = cv2line(image2, 60, 67, crop_pts2)
                if crop_pts2 and crop_pts1:
                    is_pt = False

        return self.transform(img_D1), self.transform(img_S1), self.transform(Image.fromarray(image1)), torch.tensor(
            crop_pts1), self.transform(img_D2), self.transform(img_S2), self.transform(
            Image.fromarray(image2)), torch.tensor(crop_pts2)