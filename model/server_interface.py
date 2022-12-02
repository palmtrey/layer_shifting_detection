import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from network import ResNetClassifier
from dataloader import AutomationDataset

DEVICE = 0
IMAGE_FOLDER = './images'
WEIGHTS = '/home/campalme/layer_shifting_detection/model/logs/lightning_logs/version_11/checkpoints/epoch=6-step=6272.ckpt'

if __name__ == '__main__':

    model = ResNetClassifier(resnet_version=18).cuda(DEVICE)

    model = model.load_from_checkpoint(checkpoint_path=WEIGHTS, resnet_version=18)
    model.eval()

    predict_data = AutomationDataset(folder, 'train', predict=True)

    predict_loader = DataLoader(dataset=predict_data,
                          batch_size = 1,
                          num_workers = 8,
                          shuffle = False)

    trainer = pl.Trainer(devices=[DEVICE], accelerator='cuda')
    results = trainer.predict(model=model, dataloaders=predict_loader)
    


    while True:
        images = os.listdir(IMAGE_FOLDER)

        if len(images) != 0:
            for image in images:
                predict_data = AutomationDataset(os.path.join(IMAGE_FOLDER, image), 'test', predict=2)

                predict_loader = DataLoader(dataset=predict_data,
                          batch_size = 1,
                          num_workers = 1,
                          shuffle = False)

                results = trainer.predict(model=model, dataloaders=predict_loader)
                print(results)

                os.system('rm ' + str(os.path.join(IMAGE_FOLDER, image)))
