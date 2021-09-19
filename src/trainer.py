import os
import time
from datetime import datetime
import random

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from torchvision import models,transforms
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader 
from sklearn.model_selection import KFold

from utils  import iterate
import config
from network import Vgg16,Resnet18
from Dataset import load_dataloader

import csv

# import pandas as pd


torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')


def sigmoid(x):
    return 1/(1+np.exp(-x))

class Trainer():
    def __init__(self, c):
        self.dataloaders = {}
        self.search = c
        now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        self.log_path = os.path.join(config.LOG_DIR_PATH,
                                str(now))
        os.makedirs(self.log_path, exist_ok=True)

        with open(self.log_path + "/log.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerow(['-'*20 + 'Log File' + '-'*20])

        

    def run(self):
        #実行時間計測とauc代入準備
        start = time.time()
        max_auc = -1000.0

        #CSVファイルヘッダー記述
        with open(self.log_path + "/log.csv",'a') as f:
            writer = csv.writer(f)
            writer.writerow(['model_name','lr','seed','epoch','phase','total_loss','auc'])

        for c,param in iterate(self.search):
            self.c = c
            random.seed(self.c['seed'])
            torch.manual_seed(self.c['seed'])

            if self.c['model_name'] == 'Vgg16':
                self.net = Vgg16().to(device)
            elif self.c['model_name'] == 'Resnet18':
                self.net = Resnet18().to(device)

            #params_to_update = []
            #update_param_names = ["net.classifier.6.weight","net.classifier.6.bias"]

            #for name,param in self.net.named_parameters():
            #    if name in update_param_names:
            #        param.requires_grad = True
            #        params_to_update.append(param)
            #    else:
            #        param.requires_grad = False

            self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
            self.criterion = nn.BCEWithLogitsLoss()
            self.net = nn.DataParallel(self.net)

            #K分割交差検証による実装
            self.dataset = load_dataloader(self.c['bs'])
            kf = KFold()

            cv = 0


            for learning_index,valid_index in kf.split(self.dataset['train']):
                learning_dataset = Subset(self.dataset['train'],learning_index)
                print(learning_index)
                self.dataloaders['learning'] = DataLoader(learning_dataset,self.c['bs'],
                shuffle=True,num_workers=os.cpu_count())
                valid_dataset = Subset(self.dataset['train'],valid_index)
                print(valid_index)
                self.dataloaders['valid'] = DataLoader(valid_dataset,self.c['bs'],
                shuffle=True,num_workers=os.cpu_count())
                cv = 0
                loss_sum = 0

                for epoch in range(1, self.c['n_epoch']+1):
                    self.execute_epoch(epoch, 'learning')
                    auc,loss = self.execute_epoch(epoch, 'valid')
                    loss_sum += loss
                
                    if max_auc < auc:
                        max_auc = auc
                        self.bestparam = self.c
                        self.bestparam['epoch'] = epoch
                        self.bestparam['auc'] = auc
                cv += loss_sum / kf.n_splits
            print(cv)
            
        print(self.bestparam)
        result_best = [self.bestparam['model_name'],self.bestparam['lr'],self.bestparam['seed'],self.bestparam['n_epoch'],self.bestparam['auc']]
        #df = pd.DataFrame(result_best,columns=['mode_name','lr','seed','n_epoch','auc'])
        #df.to_csv(self.log_path + '/log.csv',mode='a')

        with open(self.log_path + "/log.csv",'a') as f:
            writer = csv.writer(f)
            writer.writerow(['-'*20 + 'bestparameter' + '-'*20])
            writer.writerow(['model_name','lr','seed','n_epoch','auc'])
            writer.writerow(result_best)

        elapsed_time = time.time() - start
        print(f"実行時間 : {elapsed_time:.01f}")
            #訓練後、モデルをセーブする。
            #model_save_path = os.path.join(config.MODEL_DIR_PATH,'model.pkl')
            #torch.save(self.net.state_dict(),model_save_path)

        
        #動作かくにんのための処理 後処理だけでまとめて別ファイルにする。
        '''
        img_file_path = "./data/ChestXray001.jpg"
        img = Image.open(img_file_path)

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        transformed_img = transform(img)
        inp = transformed_img.unsqueeze_(0)
        out = self.net(inp.to(device))
        pred = out.detach().cpu().numpy()
        
        label = sigmoid(pred)[0]    
        threshold = 0.5
        if label < threshold:
            label = 0
        else:
            label = 1
        '''

        '''
        if label == 0 :
            print("This is a non-disease picture")
        else:
            print("This is a disease picture")
        '''


    def execute_epoch(self, epoch, phase):
        preds, labels, total_loss = [], [], 0
        if phase == 'learning':
            self.net.train()
        else:
            self.net.eval()

        for inputs_, labels_ in tqdm(self.dataloaders[phase]):
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)
            print(inputs_.shape,labels_.shape)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'learning'):
                outputs_ = self.net(inputs_)
                loss = self.criterion(outputs_, labels_)
                print(loss)

                if phase == 'learning':
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

            preds += [outputs_.detach().cpu().numpy()]
            labels += [labels_.detach().cpu().numpy()]

            total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        total_loss /= len(preds)

        auc = roc_auc_score(labels, preds)

        print(
            f'epoch: {epoch} phase: {phase} loss: {total_loss:.3f} auc: {auc:.3f}')
        
        result_list = [self.c['model_name'],self.c['lr'],self.c['seed'],epoch,phase,total_loss,auc]
        # df = pd.DataFrame(result_list,columns=['lr','seed','epoch','phase','total_loss','auc'])

        #if epoch == 1 and phase == 'train':
            # df.to_csv(self.log_path + '/log.csv',mode='a') 
        #    with open(self.log_path + "/log.csv",'a') as f:
        #        writer = csv.writer(f)
        #        writer.writerow(['model_name','lr','seed','epoch','phase','total_loss','auc'])
        

        with open(self.log_path + "/log.csv",'a') as f:
            writer = csv.writer(f)
            writer.writerow(result_list)

        
        # else:
            # df.to_csv(self.log_path + '/log.csv',mode='a',header=False)

        '''
        with open(self.log_path + "/log.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerow('-------Log File------')
        '''

        return auc,total_loss