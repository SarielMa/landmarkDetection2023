# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import json
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk
from tqdm import tqdm
import torch

from utils.cldetection_utils import load_train_stack_data, remove_zero_padding, check_and_make_dir


def extract_one_image_landmarks(all_gt_dict: dict, image_id: int) -> dict:
    """
    function to extract landmark information corresponding to an image
    :param all_gt_dict: a dict loaded from the train_gt.json file
    :param image_id: image id between 0 and 400
    :return: a dict containing pixel spacing and coordinates of 38 landmarks
    """
    image_dict = {'image_id': image_id}
    num_landmarks = 0
    for landmark in all_gt_dict['points']:
        point = landmark['point']
        if point[-1] != image_id:
            continue
        image_dict['scale'] = float(landmark['scale'])
        image_dict['landmark_%s' % landmark['name']] = point[:2]
        num_landmarks += 1
    image_dict['num_landmarks'] = num_landmarks
    return image_dict


def save_landmarks_list_as_csv(image_landmarks_list: list, save_csv_path: str, image_dir_path: str, image_suffix: str):
    """
    function to save the landmarks list corresponding to different images in a csv file
    :param image_landmarks_list: a list of landmark annotations, each element is an annotation of an image
    :param save_csv_path: csv file save path
    :return: None
    """
    # CSV header
    columns = ['file', 'scale']
    for i in range(38):
        columns.extend(['p{}x'.format(i + 1), 'p{}y'.format(i + 1)])
    df = pd.DataFrame(columns=columns)
    # CSV content
    for landmark in tqdm(image_landmarks_list):
        row_line = [os.path.join(image_dir_path, str(landmark['image_id']) + image_suffix), landmark['scale']]
        assert landmark['num_landmarks'] == 38
        for i in range(landmark['num_landmarks']):
            point = landmark['landmark_%s' % (i + 1)]
            row_line.extend([point[0], point[1]])
        df.loc[len(df.index)] = row_line
    # CSV writer
    df.to_csv(save_csv_path, index=False)


if __name__ == '__main__':
    # load the train_stack.mha data file using SimpleITK package
    
    # TODO: Please remember to modify it to the data file path on your computer.
    mha_file_path = './dataset/images/train_stack.mha'
    train_stack_array = load_train_stack_data(mha_file_path)

    # The function of the following script is to remove the redundant 0 padding problem.
    # Don't worry, this operation will not affect the processing of the label points of the key points,
    # because the coordinates of the key points are all in the upper left corner as the reference system
    # TODO: Please remember to modify it to the save dir path on your computer
    save_dir_path = './dataset/images/processed_images'
    check_and_make_dir(save_dir_path)
    for i in range(np.shape(train_stack_array)[0]):
        image_array = train_stack_array[i, :, :, :]
        image_array = remove_zero_padding(image_array)
        pillow_image = Image.fromarray(image_array)
        pillow_image.save(os.path.join(save_dir_path, '%s.bmp' % (i + 1)))

    # load the train_gt.json annotation file using json package
    # TODO: Please remember to modify it to the json file path on your computer
    json_file_path = './dataset/labels/train-gt.json'
    with open(json_file_path, mode='r', encoding='utf-8') as f:
        train_gt_dict = json.load(f)

    # parse out the landmark annotations corresponding to each image
    all_image_landmarks_list = []
    for i in tqdm(range(400)):
        image_landmarks = extract_one_image_landmarks(all_gt_dict=train_gt_dict, image_id=i+1)
        all_image_landmarks_list.append(image_landmarks)

    # shuffle the order of the landmark annotations list
    random.seed(2023)
    random.shuffle(all_image_landmarks_list)

    # split the training set, validation set and test set, and save them as csv files
    train_csv_path = os.path.join(os.path.dirname(save_dir_path), 'train.csv')
    print('Train CSV Path:', train_csv_path)
    save_landmarks_list_as_csv(image_landmarks_list=all_image_landmarks_list[:300],
                               save_csv_path=train_csv_path,
                               image_dir_path=save_dir_path,
                               image_suffix='.bmp')

    valid_csv_path = os.path.join(os.path.dirname(save_dir_path), 'valid.csv')
    print('Valid CSV Path:', valid_csv_path)
    save_landmarks_list_as_csv(image_landmarks_list=all_image_landmarks_list[300:350],
                               save_csv_path=valid_csv_path,
                               image_dir_path=save_dir_path,
                               image_suffix='.bmp')

    test_csv_path = os.path.join(os.path.dirname(save_dir_path), 'test.csv')
    print('Test CSV Path:', test_csv_path)
    save_landmarks_list_as_csv(image_landmarks_list=all_image_landmarks_list[350:400],
                               save_csv_path=test_csv_path,
                               image_dir_path=save_dir_path,
                               image_suffix='.bmp')
    
    # template generation
    template_path = os.path.join(os.path.dirname(os.path.dirname(save_dir_path)), 'template.pt')
    template_landmarks = None
    landmarks_frame = pd.read_csv(train_csv_path)
    for index in range(0, 300):
        if template_landmarks is None:
            template_landmarks = landmarks_frame.iloc[index, 2:].values.astype('float').reshape(-1, 2)
        else:
            template_landmarks += landmarks_frame.iloc[index, 2:].values.astype('float').reshape(-1, 2)
    template_landmarks = template_landmarks / 300
    torch.save(template_landmarks, template_path)

    