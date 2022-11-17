import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import torchvision
from torch.utils.data import Dataset, DataLoader

model = torchvision.models.resnet18(pretrained=True)
