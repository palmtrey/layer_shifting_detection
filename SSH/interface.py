import os, time
from torchvision import models
import torch
from torchvision.models import resnet

import utils

# Takes inputs from OctoPrint and CNN
# Currently being set as constants until this is figured out
shift_detected = False
printCompletion = 0.0
printAction = "RUNNING..."

IMG_DIR = '.'
CENTER = (1556.4, 1542.8) 


def display():
    os.system('cls||clear')
    if shift_detected:
        print("Shift detected.")
    else:
        print("No shift detected.")
    print("Print Completion: " + str(printCompletion) + "%")
    print("Print Action: " + printAction)
    print("\n\nCtrl+C to end")


if __name__ == '__main__':

    # Create model
    model = models.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.eval()



    # Main loop
    while True:
        display()
        time.sleep(5)

        # Capture an image
        img_path = "img.jpg"
        os.system('libcamera-jpeg -o ' + img_path + '.jpg -t 10 --width 3280 --height 2464 --gain 4 > /dev/null 2>&1')
        
        shift_detected = bool(utils.predict(model, os.path.join(IMG_DIR, img_path), CENTER))

        if shift_detected:
            stopPrint()

        if printCompletion == 100:
            printAction = "Completed"
            break
        if shift_detected:
            printAction = "Paused"
            break

def stopPrint():
    print("STOP")
