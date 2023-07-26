# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import tqdm
import json
import torch
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import io as sk_io
from skimage import transform as sk_transform

import warnings
warnings.filterwarnings('ignore')

from utils.model import ShapeDeformer
from utils.cldetection_utils import load_train_stack_data, remove_zero_padding

def main(config):
    # GPU device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    gpu_id = config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # load model
    model_args = {'patch_size': [4, 4],
                  'n_layers'  : 2,
                  'n_heads'   : 16,
                  'embed_dim' : 512,
                  'alpha'     : 100,
                  'attn_drop' : 0,
                  'C_in'      : 128,
                  'H_in'      : 128,
                  'W_in'      : 128,
                  'beta'      : 1
                  }
    model = ShapeDeformer(model_args)
    model.load_state_dict(torch.load(config.load_weight_path, map_location=device))
    model = model.to(device)

    # load test.csv
    stacked_image_array = load_train_stack_data(config.load_mha_path)

    # test result dict
    all_images_predict_landmarks_list = []
    
    # extension
    template = torch.tensor(data=torch.load('./dataset/template.pt', map_location='cpu'), dtype=torch.float32)
    template = template.view(1, -1, 2)
    template = template.to(device)

    # test mode
    with torch.no_grad():
        model.eval()
        for i in range(np.shape(stacked_image_array)[0]):
            # one image array
            image = np.array(stacked_image_array[i, :, :, :])

            # remove zero padding
            image = remove_zero_padding(image)
            height, width = np.shape(image)[:2]

            # resize
            scaled_image = sk_transform.resize(image, (512, 512), mode='constant', preserve_range=False)

            # transpose channel and add batch-size channel
            transpose_image = np.transpose(scaled_image, (2, 0, 1))
            torch_image = torch.from_numpy(transpose_image[np.newaxis, :, :, :]).float().to(device)
            landmarks_list = []
            
            # model predict
            # stage 0
            obj_pred_0 = template     
            for n_iter in range(0, 1):
                obj_deformed = model(obj_init=obj_pred_0, template=template, img=torch_image)
                u_0 = (obj_deformed - obj_pred_0).mean(dim=1, keepdim=True)
                obj_pred_0 = obj_pred_0 + u_0            
            # stage 1
            obj_pred_1 = obj_pred_0
            for n_iter in range(0, 1):
                obj_pred_1 = model(obj_init=obj_pred_1, template=template, img=torch_image)
            obj_pred = obj_pred_1 # (num_nodes, 2)
            
            for obj_iter in range(obj_pred[0].shape[0]):
                landmarks_list.append([obj_pred[0][obj_iter][0].item(), obj_pred[0][obj_iter][1].item()])
            all_images_predict_landmarks_list.append(landmarks_list)

    # save as expected format JSON file
    json_dict = {'name': 'Orthodontic landmarks', 'type': 'Multiple points'}

    all_predict_points_list = []
    for image_id, predict_landmarks in enumerate(all_images_predict_landmarks_list):
        for landmark_id, landmark in enumerate(predict_landmarks):
            points = {'name': str(landmark_id + 1),
                      'point': [landmark[0], landmark[1], image_id + 1]}
            all_predict_points_list.append(points)
    json_dict['points'] = all_predict_points_list

    # version information
    major = 1
    minor = 0
    json_dict['version'] = {'major': major, 'minor': minor}

    # JSON dict to JSON string
    json_string = json.dumps(json_dict, indent=4)
    with open(config.save_json_path, "w") as f:
        f.write(json_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--load_mha_path', type=str, default='./reference/step5_docker_and_upload/test/stack1.mha')
    parser.add_argument('--save_json_path', type=str, default='./reference/step5_docker_and_upload/test/ours_expected_output.json')

    # model load dir path
    parser.add_argument('--load_weight_path', type=str, default='./saves/best_model.pt')

    # model hyper-parameters: image_width and image_height
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)

    # model test hyper-parameters
    parser.add_argument('--cuda_id', type=int, default=0)

    experiment_config = parser.parse_args()
    main(experiment_config)
