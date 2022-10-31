import json
from PIL import Image
import numpy as np
import os
import torch
from torchvision import models, transforms
from torchvision.models import resnet
from tqdm import tqdm

from layer_shifting_utils.utils import predict

TEST_DIR_PATH = '../_data/correct_phil_ECEender3v2_cam0_Oct31_0' 
MODEL_WEIGHTS_PATH = 'trained_models/trained_model_phase2_reg.pickle'
IMAGE_EXTENSION = '.jpg'
CORRECT = True # Set this to true if the instance being assessed does not contain an error


files = os.listdir(TEST_DIR_PATH)

if not CORRECT:
    json_file = [x for x in files if x.endswith('.json')][0]
    with open(os.path.join(TEST_DIR_PATH, json_file)) as f:
        img_divide = json.load(f)






print('Test directory: ' + TEST_DIR_PATH)


model = models.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
model.eval()



# 0 denotes no shift, 1 denotes shift
results = []
labels = []

dir = os.listdir(TEST_DIR_PATH)
dir.sort()


print('\nFetching labels...')
shift = False
for img_path in tqdm(dir):
    if CORRECT:
        labels.append(0)
    elif shift == False:
        if img_path != img_divide:
            labels.append(0)
        else:
            labels.append(1)
            shift = True
    else:
        labels.append(1)

print('\nRunning tests...')
for img_path in tqdm(dir):

    if img_path.endswith(IMAGE_EXTENSION):
        results.append(predict(model, os.path.join(TEST_DIR_PATH, img_path)))


difference = [True if x == y else False for x,y in zip(results, labels)]

result = difference.count(True)/len(difference)

print('\nModel Accuracy: ' + str(result))


