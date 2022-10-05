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

DATA_PATH = 'correct_sept29_0/'
OUTPUT_PATH = 'correct_sept29_0_aug/'
CROP_BOUNDARIES = (400, 400)
FINAL_CROP = 350                     # Final size of square crop in pixels
OUTPUT_SIZE = (224, 224)
NUM_ROTATIONS = 10
NUM_TRANSLATIONS = 10


if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

for file in tqdm(os.listdir(DATA_PATH)):
    # Seed torch for randomization
    torch.manual_seed(time.time())

    # Load original image
    orig_img = Image.open(Path(DATA_PATH, file))

    file = file.split('.')[0]

    for trans in range(NUM_TRANSLATIONS):
        # Crop original image and save
        crop_orig = T.CenterCrop(CROP_BOUNDARIES)(T.ToTensor()(orig_img))
        crop_orig = T.RandomCrop(350)(crop_orig)

        # Resize the cropped image
        crop_orig = T.Resize(OUTPUT_SIZE)(crop_orig)

        # Convert to Pillow
        crop_orig = T.ToPILImage()(crop_orig)
        
        # Save resulting image
        crop_orig.save(OUTPUT_PATH + str(file) + '_crop_orig' + str(trans) + '.jpg')


    for rot in range(NUM_ROTATIONS):
        # Rotate original image
        rotated_img = T.RandomRotation(degrees=(-10, 10))(T.ToTensor()(orig_img))

        for trans in range(NUM_TRANSLATIONS):
            # Crop the rotated image
            cropped = T.CenterCrop(CROP_BOUNDARIES)(rotated_img)
            cropped = T.RandomCrop(350)(cropped)

            # Resize the cropped image
            cropped = T.Resize(OUTPUT_SIZE)(cropped)

            # Convert to Pillow
            cropped = T.ToPILImage()(cropped)

            # Save resulting image
            cropped.save(OUTPUT_PATH + str(file) + '_r' + str(rot) + '_t' + str(trans) + '.jpg')
            # break
        # break
    # break




# padding = (left, top, right, bottom)

cropped_img = T.CenterCrop(CROP_BOUNDARIES)(T.ToTensor()(orig_img))
cropped_img = T.RandomCrop(350)(cropped_img)


cropped_img = T.ToPILImage()(cropped_img)
cropped_img.save('cropped_img.png')

# f, (ax1, ax2) = plt.subplots(1, 2)

# # ax1.imshow(orig_img)
# ax1.set_xlabel('Original Image')

# ax2.imshow(cropped_img)
# ax2.set_xlabel('Rotated Image')

# plt.savefig('out.png')