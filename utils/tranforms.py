# !/usr/bin/env python
# -*- coding:utf-8 -*-


import random
import numpy as np
from skimage import transform as sk_transform


class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    Args:
        output_size tuple: Desired output size. The output is matched to output_size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = int(self.output_size[0]), int(self.output_size[1])

        image = sk_transform.resize(image, (new_h, new_w), mode='constant', preserve_range=False)
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': image, 'landmarks': landmarks}


class RandomHorizontalFlip(object):
    """
    Flip randomly the image in a sample.
    Args:
        p float: the probability of using horizontal flip augmentation
    """

    def __init__(self, p):
        assert isinstance(p, float)
        self.prob = p

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        if random.random() < self.prob:
            _, w = image.shape[:2]
            landmarks[:, 0] = w - landmarks[:, 0]
            image = image[:, ::-1, :].copy()

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """
    Convert image array in sample to Tensors.
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because numpy image: H x W x C but torch image: C X H X W
        # Please note that landmark is not normalized from original size to 512 x 512
        image = image.transpose((2, 0, 1))

        return image, landmarks

