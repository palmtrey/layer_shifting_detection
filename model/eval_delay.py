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
from network import ResNetClassifier
from dataloader import AutomationDataset
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from layer_shifting_utils.utils import predict

SPLIT_FILE = '/home/campalme/layer_shifting_detection/model/split_file7030.json'
DEVICE = 0

def calc_delay(array: list, shift_enc) -> int:
    '''Finds the first index where two of the same values exist in a list.'''

    old_value = None
    for idx, val in enumerate(array):
        if old_value == shift_enc and val == shift_enc:
            return idx - 1
        old_value = val

    return -1


def count_false_preds(array: list, shift_enc) -> int:
    '''Counts the number of times a false prediction occurs'''
    
    false_preds = 0

    old_value = None
    for idx, val in enumerate(array):
        if old_value == shift_enc and val == shift_enc:
            false_preds += 1
        old_value = val

    return false_preds


def eval_delay(folders: str, model_weights: str, image_ext: str, output_file: str) -> float:
    '''Evaluates the layer shift detection delay of a model.

    Opens a data folder data_path containing print instance
    subfolders. Each of these subfolders should contain a
    .json file with the image name where the layer shift
    first begins. The function returns a float corresponding
    to the average number of images of delay the model has.

    Delay is defined as: the number of images after a bona
    fide shift the model takes to recognize a layer shift.
    "Recognition" means two subsequent images are labeled
    as a layer shift.

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
    model = ResNetClassifier(resnet_version=18).cuda(DEVICE)

    model = model.load_from_checkpoint(checkpoint_path=model_weights, resnet_version=18)
    model.eval()

    # folders = os.listdir(data_path)
    folders.sort(key=lambda folder: int(folder.split('_')[-1]))
    accuracies = []
    delays = []
    all_false_preds = []


    for idx, folder in enumerate(folders):
        print('\n(' + str(idx + 1) + '/' + str(len(folders)) + ') Folder: ' + str(folder))
        files = os.listdir(folder)
        images = [x for x in files if x.endswith(image_ext)]
        images.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

        # print(images)

        try:
            with open(os.path.join(folder, '_shift.json')) as f:
                split_name = json.load(f)

            with open(os.path.join(folder, '_center.json')) as f:
                center = json.load(f)

            # 0 denotes no shift, 1 denotes shift
            results = []
            labels = []

            print('\nFetching labels...')
            shift = False

            for img_path in tqdm(images):
                if shift == False:
                    if img_path.replace(':', '_') != split_name:
                        labels.append(0)
                    else:
                        labels.append(1)
                        shift_start = len(labels) - 1
                        shift = True
                else:
                    labels.append(1)

            print('\nRunning tests...')
            # for img_path in tqdm(images):

            #     if img_path.endswith(image_ext):
            #         results.append(predict(model, os.path.join(folder, img_path), center))

            predict_data = AutomationDataset(folder, 'train', predict=True)

            predict_loader = DataLoader(dataset=predict_data,
                          batch_size = 1,
                          num_workers = 8,
                          shuffle = False)

            trainer = pl.Trainer(devices=[DEVICE], accelerator='cuda')
            results = trainer.predict(model=model, dataloaders=predict_loader)

            # print(results)


            difference = ['' if x == y else 'x' for x,y in zip(results, labels)]

            accuracy = difference.count('')/len(difference)
            delay = calc_delay(difference[shift_start:], '')
            false_preds = count_false_preds(difference[0:shift_start], '')

            accuracies.append(accuracy)
            delays.append(delay)
            all_false_preds.append(false_preds)

            print('\nResults for ' + str(folder) + '\n' + '-'*10)
            print('Model Accuracy: ' + str(accuracy))
            print('Model Delay: ' + str(delay))
            print('False Predictions: ' + str(false_preds))
        
            

            print('Bona Fide Labels: ' + str(labels))
            print('Predictions: ' + str(results))
            # print('')
            # print('Difference: ' + str(difference[shift_start:]))

            # print(calc_delay(difference[shift_start:], ''))

            




            # break
        except FileNotFoundError:
            print('json file missing or this is a correct folder. Skipping...')
        
    accuracy_avg = sum(accuracies)/len(accuracies)
    relevant_delays = [x for x in delays if x != -1]
    delay_avg = sum(relevant_delays)/len(relevant_delays)

    print('\n\nAverage Accuracy: ' + str(accuracy_avg))
    print('Average Delay: ' + str(delay_avg))

    out = {
            'folders':folders,
            'accuracies':accuracies,
            'delays':delays,
            'accuracy_avg':accuracy_avg,
            'delay_avg':delay_avg
        }

    with open(output_file, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    with open('split_file7030.json', 'r') as f:
        split = json.load(f)

    WEIGHTS = '/home/campalme/layer_shifting_detection/model/logs/lightning_logs/version_7/checkpoints/epoch=99-step=89600.ckpt'

    eval_delay(split['train'], WEIGHTS, '.jpg', 'results_val.json')