from ctypes import resize
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import numpy as np
import torchvision.transforms as T

data_path = 'correct_sept29_0'

plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open(Path(data_path, '2022-09-29_13_47_44.jpg'))
torch.manual_seed(0)



rotated_img = T.RandomRotation(degrees=360)(T.ToTensor()(orig_img))

f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)

ax1.imshow(orig_img)
ax1.set_xlabel('Original Image')

ax2.imshow(T.ToPILImage()(rotated_img))
ax2.set_xlabel('Rotated Image')

plt.savefig('out.png')