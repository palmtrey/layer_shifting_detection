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
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


import wandb

DEVICE = 1
BATCH_SIZE = 10
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 5e-3
EPOCHS = 100
SPLIT_FILE = '/home/campalme/layer_shifting_detection/model/split_file7030.json'
OPTIMIZER = 'sgd'

tb_logger = TensorBoardLogger(save_dir="logs/")
wandb_logger = WandbLogger(project="layer_shifting_detection")



model = ResNetClassifier(num_classes = 2, resnet_version = 18, batch_size=BATCH_SIZE, epochs=EPOCHS,
                            optimizer = OPTIMIZER, lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY, tune_fc_only=False).cuda(DEVICE)

# model = model.load_from_checkpoint(checkpoint_path='/home/campalme/layer_shifting_detection/model/lightning_logs/version_2/checkpoints/epoch=0-step=100.ckpt', num_classes=2, resnet_version=18)

train_data = AutomationDataset(SPLIT_FILE, 'train')

train_loader = DataLoader(dataset=train_data,
                          batch_size = BATCH_SIZE,
                          num_workers = 8,
                          shuffle = True)

test_data = AutomationDataset(SPLIT_FILE, 'test')

test_loader = DataLoader(dataset=test_data,
                         batch_size = BATCH_SIZE,
                         num_workers = 8,
                         shuffle = True)

trainer = pl.Trainer(max_epochs=EPOCHS, devices=[DEVICE], accelerator='cuda', logger=[tb_logger ,wandb_logger])
trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=test_loader)

# trainer.validate(model, dataloaders=test_loader)