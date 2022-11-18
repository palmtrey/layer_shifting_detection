import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import torchvision
from torch.utils.data import Dataset, DataLoader
from dataloader import AutomationDataset
from torchvision.models.resnet import ResNet18_Weights

from network import ResNetClassifier

DEVICE = 0
BATCH_SIZE = 64

model = ResNetClassifier(num_classes = 2, resnet_version = 18,
                            optimizer = 'adam', lr = 1e-3).cuda(DEVICE)

# model = model.load_from_checkpoint(checkpoint_path='/home/campalme/layer_shifting_detection/model/lightning_logs/version_2/checkpoints/epoch=0-step=100.ckpt', num_classes=2, resnet_version=18)

train_data = AutomationDataset('/home/campalme/layer_shifting_detection/model/split_file.json', 
                                'train')

train_loader = DataLoader(dataset=train_data,
                          batch_size = BATCH_SIZE,
                          num_workers = 8,
                          shuffle = True)

test_data = AutomationDataset('/home/campalme/layer_shifting_detection/model/split_file.json',
                              'test')

test_loader = DataLoader(dataset=test_data,
                         batch_size = BATCH_SIZE,
                         num_workers = 8,
                         shuffle = True)

trainer = pl.Trainer(limit_train_batches=100, devices=[DEVICE], accelerator='cuda')
# trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=test_loader)

trainer.validate(model, dataloaders=test_loader)