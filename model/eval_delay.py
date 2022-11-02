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

def calc_delay(array: list, shift_enc) -> int:
    '''Finds the first index where two of the same values exist in a list.'''

    old_value = None
    for idx, val in enumerate(array):
        if old_value == shift_enc and val == shift_enc:
            return idx - 1
        old_value = val

    return -1


def eval_delay(data_path: str, model_weights: str, image_ext: str, output_file: str) -> float:
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
    model = models.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
    model.eval()


    folders = os.listdir(data_path)
    folders.sort(key=lambda folder: int(folder.split('_')[-1]))
    accuracies = []
    delays = []


    for idx, folder in enumerate(folders):
        print('\n(' + str(idx + 1) + '/' + str(len(folders)) + ') Folder: ' + str(folder))
        files = os.listdir(os.path.join(data_path, folder))
        images = [x for x in files if x.endswith(image_ext)]
        images.sort()

        try:
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
                    if img_path.replace(':', '_') != split_name:
                        labels.append(0)
                    else:
                        labels.append(1)
                        shift_start = len(labels) - 1
                        shift = True
                else:
                    labels.append(1)

            print('\nRunning tests...')
            for img_path in tqdm(images):

                if img_path.endswith(image_ext):
                    results.append(predict(model, os.path.join(data_path, folder, img_path)))


            difference = ['' if x == y else 'x' for x,y in zip(results, labels)]

            accuracy = difference.count('')/len(difference)
            delay = calc_delay(difference[shift_start:], '')

            accuracies.append(accuracy)
            delays.append(delay)

            print('\nResults for ' + str(folder) + '\n' + '-'*10)
            print('Model Accuracy: ' + str(accuracy))
            print('Model Delay: ' + str(delay))
        except IndexError:
            print('json file missing for ' + str(folder) + '. Skipping...')

        

        # print('Bona Fide Labels: ' + str(labels[shift_start:]))
        # print('Predictions: ' + str(results[shift_start:]))
        # print('')
        # print('Difference: ' + str(difference[shift_start:]))

        # print(calc_delay(difference[shift_start:], ''))

        




        # break

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
    eval_delay('../_data/temp', '../model/trained_models/trained_model_phase2_reg.pickle', '.jpg', 'results_val.json')