# Title: eval_delay.py
# Purpose: To evaluate the delay between when a layer shift occurs
#          and when the CNN model detects the delay.
# Author: Cameron Palmer, campalme@clarkson.edu
# Last Modified: October 18th, 2022

import json
from PIL import Image
import numpy as np
import os
import torch
from torchvision import models, transforms
from torchvision.models import resnet
from tqdm import tqdm

from layer_shifting_utils.utils import predict

def eval_delay(data_path: str, model_weights: str, image_ext: str) -> float:
    '''Evaluates the layer shift detection delay of a model.

    Opens a data folder data_path containing print instance
    subfolders. Each of these subfolders should contain a
    .json file with the image name where the layer shift
    first begins. The function returns a float corresponding
    to the average number of images of delay the model has.

    Args:
        data_path: A path to a folder containing layer shift
            print instance subfolders.
        model_weights: A path to the .pickle file containing
            model weights. Model is assumed to be ResNet18.
        image_ext: Extension for images. Ex. '.jpg'

    Returns:
        A float value corresponding to the average number of
        images it takes after the actual layer shift has 
        occured for the model to detect the shift.
    '''

    
    # Load the model
    model = models.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
    model.eval()


    
    

    folders = os.listdir(data_path)

    for folder in folders:
        files = os.listdir(os.path.join(data_path, folder))
        images = [x for x in files if x.endswith(image_ext)]
        images.sort()
        json_file = [x for x in files if x.endswith('.json')][0]

        with open(os.path.join(data_path, folder, json_file)) as f:
            split_name = json.load(f)

        # 0 denotes no shift, 1 denotes shift
        results = []
        labels = []

        print('\nFetching labels...')
        shift = False
        for img_path in tqdm(images):
            if shift == False:
                if img_path != split_name:
                    labels.append(0)
                else:
                    labels.append(1)
                    shift = True
            else:
                labels.append(1)

        print('\nRunning tests...')
        for img_path in tqdm(images):

            if img_path.endswith(image_ext):
                results.append(predict(model, os.path.join(TEST_DIR_PATH, img_path)))


        difference = [True if x == y else False for x,y in zip(results, labels)]

        result = difference.count(True)/len(difference)

        print('\nModel Accuracy: ' + str(result))






        break

if __name__ == '__main__':
    eval_delay('../data/', 'trained_model_phase1.pickle', '.jpg')