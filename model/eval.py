from PIL import Image
import numpy as np
import os
import torch
from torchvision import models, transforms
from torchvision.models import resnet
from tqdm import tqdm

TEST_DIR_PATH = 'data/shift_oct3_11/' 


# For oct3_9
# IMG_DIVIDE = '2022-10-03_16:23:00.jpg'

# For oct3_10
# IMG_DIVIDE = '2022-10-03_16:58:58.jpg'  # Name of image where the layer shift can be seen for the first time

# For oct3_11
IMG_DIVIDE = '2022-10-03_17:33:57.jpg'

# For oct3_11_aug
# IMG_DIVIDE = '2022-10-03_17:33:57_crop_orig0.jpg'

# For oct4_0
# IMG_DIVIDE = '2022-10-04_13:03:03.jpg'




MODEL_WEIGHTS_PATH = 'trained_model_phase1.pickle'

CROP_BOUNDARIES = (350, 350)
FINAL_CROP = 350                     # Final size of square crop in pixels
OUTPUT_SIZE = (224, 224)

print('Test directory: ' + TEST_DIR_PATH)


model = models.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
model.eval()

data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(CROP_BOUNDARIES),
            #transforms.RandomCrop(FINAL_CROP),
            transforms.Resize(OUTPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

# 0 denotes no shift, 1 denotes shift
results = []
labels = []

dir = os.listdir(TEST_DIR_PATH)
dir.sort()


print('\nFetching labels...')
shift = False
for img_path in tqdm(dir):
    if shift == False:
        if img_path != IMG_DIVIDE:
            labels.append(0)
        else:
            labels.append(1)
            shift = True
    else:
        labels.append(1)

print('\nRunning tests...')
for img_path in tqdm(dir):

    test_img = Image.open(TEST_DIR_PATH + img_path)

    test_img = data_transforms['val'](test_img).unsqueeze(0)

    output = model(test_img)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    results.append(np.argmax([list(probabilities)[0].item(), list(probabilities)[1].item()]))


difference = [True if x == y else False for x,y in zip(results, labels)]

result = difference.count(True)/len(difference)

print('\nModel Accuracy: ' + str(result))

