# NOTE: This file does NOT work, and should NOT be used.

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import numpy as np
import torchvision.transforms as T
import time
from tqdm import tqdm

FINAL_CROP = 350                     # Final size of square crop in pixels
OUTPUT_SIZE = (224, 224)
NUM_ROTATIONS = 10
NUM_TRANSLATIONS = 10


def augment_folder(folder_path, output_path, crop_bounds):
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    for file in tqdm(os.listdir(folder_path)):
        # Seed torch for randomization
        torch.manual_seed(time.time())

        # Load original image
        orig_img = Image.open(os.path.join(folder_path, file))

        file_name = file.split('.')[0]

        for trans in range(NUM_TRANSLATIONS):
            # Crop original image and save
            crop_orig = T.CenterCrop(crop_bounds)(T.ToTensor()(orig_img))
            crop_orig = T.RandomCrop(FINAL_CROP)(crop_orig)

            # Resize the cropped image
            crop_orig = T.Resize(OUTPUT_SIZE)(crop_orig)

            # Convert to Pillow
            crop_orig = T.ToPILImage()(crop_orig)
            
            # Save resulting image
            crop_orig.save(output_path + str(file_name) + '_crop_orig' + str(trans) + '.jpg')


        for rot in range(NUM_ROTATIONS):
            # Rotate original image
            rotated_img = T.RandomRotation(degrees=(-10, 10))(T.ToTensor()(orig_img))

            for trans in range(NUM_TRANSLATIONS):
                # Crop the rotated image
                cropped = T.CenterCrop(crop_bounds)(rotated_img)
                cropped = T.RandomCrop(FINAL_CROP)(cropped)

                # Resize the cropped image
                cropped = T.Resize(OUTPUT_SIZE)(cropped)

                # Convert to Pillow
                cropped = T.ToPILImage()(cropped)

                # Save resulting image
                cropped.save(output_path + str(file_name) + '_r' + str(rot) + '_t' + str(trans) + '.jpg')
                # break
            # break
        # break

if __name__ == '__main__':
    INPUT_FOLDER = '/media/DATACENTER2/campalme/automation_dataset/images/'
    OUTPUT_FOLDER = '/media/DATACENTER2/campalme/automation_dataset/images_augmented/'

    if not os.path.isdir(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    for folder in os.listdir(INPUT_FOLDER):
        if not os.path.isdir(os.path.join(OUTPUT_FOLDER, folder)):
            

            if os.path.isfile(os.path.join(INPUT_FOLDER, folder, '_center.json')) 


