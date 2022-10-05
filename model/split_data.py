import random
import os
import shutil
import time
from tqdm import tqdm

random.seed(time.time())

POSITIVE_DATA_PATH = 'phase_1_labeled/BAD_aug/'
NEGATIVE_DATA_PATH = 'phase_1_labeled/GOOD_aug/'

TRAIN_DATA_OUT_PATH = 'phase_1_labeled/train/'
VAL_DATA_OUT_PATH = 'phase_1_labeled/val/'

POSITIVE_LABEL = 'shift'
NEGATIVE_LABEL = 'no_shift'

if not os.path.isdir(TRAIN_DATA_OUT_PATH + POSITIVE_LABEL):
    os.makedirs(TRAIN_DATA_OUT_PATH + POSITIVE_LABEL, exist_ok=True)

if not os.path.isdir(TRAIN_DATA_OUT_PATH + NEGATIVE_LABEL):
    os.makedirs(TRAIN_DATA_OUT_PATH + NEGATIVE_LABEL, exist_ok=True)

if not os.path.isdir(VAL_DATA_OUT_PATH + POSITIVE_LABEL):
    os.makedirs(VAL_DATA_OUT_PATH + POSITIVE_LABEL, exist_ok=True)

if not os.path.isdir(VAL_DATA_OUT_PATH + NEGATIVE_LABEL):
    os.makedirs(VAL_DATA_OUT_PATH + NEGATIVE_LABEL, exist_ok=True)

positives = os.listdir(POSITIVE_DATA_PATH)
negatives = os.listdir(NEGATIVE_DATA_PATH)

random.shuffle(positives)
random.shuffle(negatives)

positives_train = positives[0:len(positives)//2]
positives_val = positives[len(positives)//2 + 1:]

negatives_train = negatives[0:len(negatives)//2]
negatives_val = negatives[len(negatives)//2 + 1:]

print('Copying positives to training folder...')
for positive in tqdm(positives_train):
    shutil.copy(POSITIVE_DATA_PATH + str(positive), TRAIN_DATA_OUT_PATH + POSITIVE_LABEL + '/' + positive)

print('Copying positives to validation folder...')
for positive in tqdm(positives_val):
    shutil.copy(POSITIVE_DATA_PATH + str(positive), VAL_DATA_OUT_PATH + POSITIVE_LABEL + '/' + positive)

print('Copying negatives to training folder...')
for negative in tqdm(negatives_train):
    shutil.copy(NEGATIVE_DATA_PATH + str(negative), TRAIN_DATA_OUT_PATH + NEGATIVE_LABEL + '/' + negative)

print('Copying negatives to validation folder...')
for negative in tqdm(negatives_val):
    shutil.copy(NEGATIVE_DATA_PATH + str(negative), VAL_DATA_OUT_PATH + NEGATIVE_LABEL + '/' + negative)





