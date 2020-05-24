# *_*coding:utf-8 *_*
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision
import cv2
import numpy as np
from PIL import Image
import random
import pickle

import src.utils as utils

class TextLineDataset(Dataset):
    def __init__(self, text_file=None, transform=None, converter=None):
        self.text_file = text_file
        with open(self.text_file, "r", encoding="utf-8") as fp:
            self.lines = fp.readlines()
            self.nSamples = len(self.lines)

        self.transform = transform
        self.converter = converter

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        line_split = self.lines[index].strip().split(' ')
        img_path = line_split[0]
        img = Image.open(img_path).convert('L')  # 已经清洗过数据

        # 图像归一化
        if self.transform is not None:
            img = self.transform(img)

        # 标签归一化
        text = line_split[1]
        label, length = self.converter.encode(text) # 返回mask和length

        return img, label, length, text

class ResizeNormalize(object):

    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height
        self.toTensor = torchvision.transforms.ToTensor()

    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape
        height = self.img_height
        width = int(w * height / h)
        if width >= self.img_width:
            img = cv2.resize(img, (self.img_width, self.img_height))
        else:
            img = cv2.resize(img, (width, height))
            img_pad = np.zeros((self.img_height, self.img_width), dtype=img.dtype)
            img_pad[:height, :width] = img
            img = img_pad
        img = Image.fromarray(img)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class AlignCollate(object):

    def __init__(self):
        pass

    def __call__(self, batch):
        images, labels, lengths, text = zip(*batch)

        images = torch.stack(images, dim=0)
        labels = torch.cat(labels)
        lengths = torch.cat(lengths)

        return images, labels, lengths, text


if __name__ == '__main__':
    # text_file = "../data/test_list.txt"
    text_file = "../data/train_text_file.txt"
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    converter = utils.strLabelConverter(alphabet)
    train_set = TextLineDataset(text_file, transform=ResizeNormalize(100, 32), converter=converter)
    # dataset
    for i in range(1):
        images, labels, lengths, text = train_set[i]
        print(images.size())
        print(labels)
        print(lengths)
        print(text)

