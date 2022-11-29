import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def predict(model: models.ResNet, input_path: str, center: tuple) -> int:
    '''Uses a loaded ResNet model and an input image and returns a prediction.
    0 - no_shift
    1 - shift
    '''

    CROP_EXTENT = 350
    OUTPUT_SIZE = (224, 224)
    FINAL_CROP = 700

    crop_boundaries = (center[0]-CROP_EXTENT, center[1]-CROP_EXTENT, center[0]+CROP_EXTENT,center[1]+CROP_EXTENT)
    # print(crop_boundaries)
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.CenterCrop(CROP_BOUNDARIES),
            transforms.Resize(OUTPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    img = Image.open(input_path)

    aug = TF.crop(T.ToTensor()(img), int(center[1] - FINAL_CROP//2), int(center[0] - FINAL_CROP//2), FINAL_CROP, FINAL_CROP)
    aug = T.Resize(OUTPUT_SIZE)(aug)
    aug = T.Normalize((0.48232,), (0.23051,))(aug).unsqueeze(0)

    # test_img = test_img.crop(crop_boundaries)

    # test_img = data_transforms['val'](test_img).unsqueeze(0)
    with torch.no_grad():
        output = model(aug)

    print(output)

    probabilities = torch.argmax(output, 1)

    print(int(probabilities[0]))

    # probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # print(probabilities)

    return int(probabilities[0])