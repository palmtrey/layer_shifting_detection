import os, time
from torchvision import models
import torch
from torchvision.models import resnet

import utils

shift_detected = False
IMG_DIR = '.'
CENTER = (1556.4, 1542.8) 

if __name__ == '__main__':

    # Create model
    model = models.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    model.load_state_dict(torch.load("128.153.28.135\oneilaj\layer_shifting_detection\model\trained_models", map_location=torch.device('cpu')))
    model.eval()



    # Main loop
    while True:

        path = '/images/unprocessed'
        while (os.stat(path).st_size != 0)
            images = os.listdir(path)
            img_path = os.path.join(path, images[0])

            shift_detected = bool(utils.predict(model, os.path.join(IMG_DIR, img_path), CENTER))

            if shift_detected:
                os.system('scp message.txt pi@128.53.134.177:/layer_shifting_detection/SSH/messages')
                stopPrint()

def stopPrint():
    print("STOP")
