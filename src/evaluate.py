import os
import time
from datetime import datetime
import random

from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import config
import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader 
from network import Vgg16,Resnet18
from Dataset import load_dataloader

c = {
    'model_name': 'Resnet18',
    'seed': [0], 'bs': 64, 'lr': [1e-4], 'n_epoch': [10]
}

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Evaluater():
    def __init__(self,c):
        self.dataloaders = {}
        self.search = c
        self.c = c
        now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        model_path = os.path.join(config.MODEL_DIR_PATH,
                                'model.pth')
        if self.c['model_name'] == 'Vgg16':
                self.net = Vgg16().to(device)
        elif self.c['model_name'] == 'Resnet18':
                self.net = Resnet18().to(device)

        self.net.load_state_dict(torch.load(model_path,map_location=device))
        self.criterion = nn.BCEWithLogitsLoss()

    def run(self):
            self.dataset = load_dataloader(self.c['bs'])
            test_dataset = self.dataset['test']
            self.dataloaders['test'] = DataLoader(test_dataset,self.c['bs'],
                    shuffle=True,num_workers=os.cpu_count())

            preds, labels,total_loss,accuracy= [], [],0,0
            right,notright = 0,0
            self.net.eval()

            for inputs_, labels_ in tqdm(self.dataloaders['test']):
                inputs_ = inputs_.to(device)
                labels_ = labels_.to(device)


                torch.set_grad_enabled(False)
                outputs_ = self.net(inputs_)
                loss = self.criterion(outputs_, labels_)
                #total_loss += loss.item()


                preds += [outputs_.detach().cpu().numpy()]
                labels += [labels_.detach().cpu().numpy()]

                total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            total_loss /= len(preds)

            worst_id = np.argmax(preds-labels)
            worst = (preds-labels).max()

            print(np.argmax(preds-labels),(preds-labels).max())

            print(preds[worst_id],labels[worst_id])

            threshold = 0.5
            right += ((preds-labels) < threshold).sum()
            notright += len(preds) - ((preds - labels) < threshold).sum()

            accuracy = right / len(test_dataset)
            mae = mean_absolute_error(preds,labels)
            mse = mean_squared_error(preds,labels)
            print('accuracy :',accuracy)
            print('MAE :',mae)
            print('AE : ',mae*len(preds))
            print('MSE',mse)
            print('SE',mse*len(preds))



if __name__ == '__main__':
    evaluater = Evaluater(c)
    evaluater.run()
