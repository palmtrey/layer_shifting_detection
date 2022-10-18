from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms

def predict(model: models.ResNet, input_path: str) -> int:
    '''Uses a loaded ResNet model and an input image and returns a prediction.
    0 - no_shift
    1 - shift
    '''

    CROP_BOUNDARIES = (350, 350)
    OUTPUT_SIZE = (224, 224)


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

    test_img = Image.open(input_path)

    test_img = data_transforms['val'](test_img).unsqueeze(0)

    output = model(test_img)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    return np.argmax([list(probabilities)[0].item(), list(probabilities)[1].item()])