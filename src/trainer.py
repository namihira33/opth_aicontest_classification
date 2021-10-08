import os
import sys
import time
from datetime import datetime
import random

from tqdm import tqdm
import numpy as np
from sklearn.metrics import *
import pickle
import tensorboardX as tbx

from torchvision import models,transforms
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader 
from sklearn.model_selection import KFold

from utils  import *
import config
from network import *
from Dataset import load_dataloader

import csv
import matplotlib.pyplot as plt
import seaborn as sns

#import pandas as pd


torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

class Trainer():
    def __init__(self, c):
        self.dataloaders = {}
        self.prms = []
        self.search = c
        self.n_seeds = len(c['seed'])
        self.n_splits = 5        
        self.loss,self.mae,self.mse,self.r_score= {},{},{},{}
        self.now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        self.log_path = os.path.join(config.LOG_DIR_PATH,
                                str(self.now))
        os.makedirs(self.log_path, exist_ok=True)
        self.tb_writer = tbx.SummaryWriter()

        with open(self.log_path + "/log.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerow(['-'*20 + 'Log File' + '-'*20])

        

    def run(self):
        #実行時間計測とmae代入準備
        start = time.time()
        n_iter = 1

        #Initialization -> Score
        for phase in ['learning','valid']:
            self.loss[phase] = 0
            self.mae[phase] = 0
            self.r_score[phase] = 0
            self.mse[phase] = 0
        #ヒートマップ描画用のリスト
        validheat,heat_index = [],[]

        #Seed平均を取るためのリスト
        seed_valid = []

        #CSVファイルヘッダー記述
        with open(self.log_path + "/log.csv",'a') as f:
            writer = csv.writer(f)
            writer.writerow(['model_name','lr','seed','epoch','phase','total_loss','mae'])

        for c,param in iterate(self.search):
            print('Parameter :',c)
            self.c = c
            random.seed(self.c['seed'])
            torch.manual_seed(self.c['seed'])

            mn = self.c['model_name']

            if mn == 'Vgg16':
                self.net = Vgg16()
            elif mn == 'Vgg16-bn':
                self.net = Vgg16_bn()
            elif mn == 'Vgg19':
                self.net = Vgg19()
            elif mn == 'Vgg19-bn':
                self.net = Vgg19_bn()
            elif mn == 'Resnet18':
                self.net = Resnet18()
            elif mn == 'Resnet34':
                self.net = Resnet34()
            elif mn == 'Resnet50':
                self.net = Resnet50()
            elif mn == 'Squeezenet':
                self.net = Squeezenet()
            elif mn == 'Densenet':
                self.net = Densenet()
            elif mn == 'Inception':
                self.net = Inception()
            elif mn == 'Mobilenet-large':
                self.net = Mobilenet_large()
            elif mn == 'Mobilenet-small':
                self.net = Mobilenet_small()
            
            self.net = self.net.to(device)
            self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
            #self.criterion = nn.BCELoss()
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.net = nn.DataParallel(self.net)

            #成績の初期化
            losses,maes = {},{}
            for phase in ['learning','valid']:
                losses[phase] = []
                maes[phase] = []

            memory = {}
            memory2 = {}
            for phase in ['learning','valid']:
                if self.c['cv'] == 0:
                    memory[phase] = [[] for x in range(1)]  #self.n_splits
                    memory2[phase] = [[] for x in range(1)] #self.n_splits
                else:
                    memory[phase] = [[] for x in range(self.n_splits)]
                    memory2[phase] = [[] for x in range(self.n_splits)]

            self.dataset = load_dataloader(self.c['bs'])
            kf = KFold(n_splits=5,shuffle=True,random_state=self.c['seed'])

            epoch_n = 1

            for a,(learning_index,valid_index) in enumerate(kf.split(self.dataset['train'])):
                #データセットが切り替わるたびに、ネットワークの重み,バイアスを初期化
                #utils.py -> init_weights()
                self.net.apply(init_weights)
                self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
                    
                learning_dataset = Subset(self.dataset['train'],learning_index)
                self.dataloaders['learning'] = DataLoader(learning_dataset,self.c['bs'],
                shuffle=True,num_workers=os.cpu_count())
                valid_dataset = Subset(self.dataset['train'],valid_index)
                self.dataloaders['valid'] = DataLoader(valid_dataset,self.c['bs'],
                shuffle=True,num_workers=os.cpu_count())


                for epoch in range(1, self.c['n_epoch']+1):

                    learningmae,learningloss,learningr_score,learningmse \
                        = self.execute_epoch(epoch, 'learning')

                    validmae,validloss,validr_score,validmse\
                        = self.execute_epoch(epoch, 'valid')
                    
                    if not self.c['evaluate']:
                        #validmae,validloss,validr_score\
                        #    = self.execute_epoch(epoch, 'valid')
                        #self.tb_writer.add_scalar('Loss/{}'.format('valid'),validloss,epoch)
                        #self.tb_writer.add_scalar('Mae/{}'.format('valid'),validmae,epoch)
                        #self.tb_writer.add_scalar('R2_score/{}'.format('valid'),validr_score,epoch)
                        memory['valid'][a].append(validmae)
                        memory2['valid'][a].append(validloss)

                        temp = validmae,epoch_n,self.c
                        self.prms.append(temp)

                    epoch_n += 1
                    if epoch == self.c['n_epoch']:
                        self.mae['learning'] += learningmae
                        self.mae['valid'] += validmae
                        self.loss['learning'] += learningloss
                        self.loss['valid'] += validloss
                        self.mse['learning'] += learningmse
                        self.mse['valid'] += validmse
                        self.r_score['learning'] += learningr_score
                        self.r_score['valid'] += validr_score
                        
                
                #n_epoch後の処理
                save_process_path = os.path.join(config.LOG_DIR_PATH,
                                str(self.now))
                if not self.c['cv']:
                    break

            #分割交差検証後の処理
            memory['learning'] = list(np.mean(memory['learning'],axis=0))
            memory2['learning'] = list(np.mean(memory2['valid'],axis=0))
            if not self.c['evaluate']:
                memory['valid'] = list(np.mean(memory['valid'],axis=0))
                memory2['valid'] = list(np.mean(memory2['valid'],axis=0))
                seed_valid.append(memory['valid'])

            if self.c['cv']:
                self.tb_writer = tbx.SummaryWriter()
                for phase in ['learning','valid']:
                    for b in range(len(memory[phase])):
                        tbx_write(self.tb_writer,b+1,phase,memory[phase][b],memory2[phase][b])
                        with open(self.log_path + "/log.csv",'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([self.c,phase,'ValidMAE',memory[phase][b]])
                    for b in range(len(memory[phase])):
                        with open(self.log_path + "/log.csv",'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([self.c,phase,'ValidLoss',memory2[phase][b]])

                    
                        
                #JSON形式でTensorboardに保存した値を残しておく。
                self.tb_writer.export_scalars_to_json('./log/all_scalars.json')
                self.tb_writer.close()

            #乱数シードiterごとに、平均を取り、これを記録。
            if not (n_iter%self.n_seeds):
                temp = self.n_seeds * self.n_splits
                for phase in ['learning','valid']:
                    self.mae[phase]  /= temp
                    self.loss[phase] /= temp
                    self.mse[phase] /= temp
                    self.r_score[phase] /= temp
                    #self.tb_writer.add_scalar('Loss/{}'.format(phase),self.loss[phase],self.c['n_epoch'])
                    #self.tb_writer.add_scalar('Mae/{}'.format(phase),self.mae[phase],self.c['n_epoch'])
                    #self.tb_writer.add_scalar('R2_score/{}'.format(phase),self.r_score[phase],self.c['n_epoch'])
                    #self.tb_writer.add_scalar('Mse/{}'.format(phase),self.mse[phase],self.c['n_epoch'])
                    #initialization -> Score
                    self.mae[phase] = 0
                    self.loss[phase] = 0
                    self.mse[phase] = 0
                    self.r_score[phase] = 0
                print(memory['valid'])
                seed_valid = np.mean(seed_valid,axis=0)
                validheat.append(memory['valid'])
                heat_index.append(self.c['lr'])
                seed_valid = []

                
            n_iter += 1

        #パラメータiter後の処理。

        best_prms = sorted(self.prms,key=lambda x:x[0])
        with open(self.log_path + "/log.csv",'a') as f:
            writer = csv.writer(f)
            writer.writerow(['-'*20 + 'bestparameter' + '-'*20])
            writer.writerow(['model_name','lr','seed','n_epoch','mae'])
            writer.writerow(best_prms[0:10])

        #学習率・10epoch経過後のヒートマップの描画
            validheat = [l[::5] for l in validheat[:]]
            print(validheat)
            fig,ax = plt.subplots(figsize=(12,8))
            xtick = list(map(lambda x:5*x-4,list(range(1,len(validheat[0])+1))))
            xtick = [str(x) + 'ep' for x in xtick]
            sns.heatmap(validheat,annot=True,cmap='Set3',fmt='.2f',
                xticklabels=xtick,yticklabels=heat_index,vmin=2.5,vmax=10,
                cbar_kws = dict(label='Valid Age MAE'))
            ax.set_ylabel('learning rate')
            ax.set_xlabel('num of epoch')
            ax.set_title('num of seeds : ' + str(self.n_seeds))
            fig.savefig('./log/images/'+self.now + 'train_ep.png')

        elapsed_time = time.time() - start
        print(f"実行時間 : {elapsed_time:.01f}")
        #訓練後、モデルをセーブする。
        #(実行回数)_(モデル名)_(学習epoch).pth で保存。
        try : 
             model_name = self.search['model_name']
             n_ep = self.search['n_epoch']
             n_ex = 0
             with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'r') as f:
                 n_ex = len(f.readlines())

             with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'a') as f:
                 writer = csv.writer(f)
                 writer.writerow([self.now,n_ex,model_name,n_ep])
             save_path = '{:0=2}'.format(n_ex)+ '_' + model_name + '_' + '{:0=3}'.format(n_ep)+'ep.pth'
             model_save_path = os.path.join(config.MODEL_DIR_PATH,save_path)
             torch.save(self.net.module.state_dict(),model_save_path)
        except FileNotFoundError:
            with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'w') as f:
                 writer = csv.writer(f)
                 writer.writerow(['Time','n_ex','Model_name','n_ep'])


        #JSON形式でTensorboardに保存した値を残しておく。
        self.tb_writer.export_scalars_to_json('./log/all_scalars.json')
        self.tb_writer.close()

    #1epochごとの処理
    def execute_epoch(self, epoch, phase):
        preds, labels,total_loss= [], [],0
        if phase == 'learning':
            self.net.train()
        else:
            self.net.eval()

        for inputs_, labels_ in tqdm(self.dataloaders[phase]):
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)
            self.optimizer.zero_grad()


            with torch.set_grad_enabled(phase == 'learning'):
                outputs_ = self.net(inputs_)
                #MSELoss,その他用コード
                delta = 1.0e-26
                loss = self.criterion((outputs_+delta).log(), labels_)
                total_loss += loss.item()

                if phase == 'learning':
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

            preds += [outputs_.detach().cpu().numpy()]
            labels += [labels_.detach().cpu().numpy()]
            total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        total_loss /= len(preds)
        
        #ハードな答えの出し方
        #preds = np.argmax(preds,1)
        #ソフトな答えの出し方 加重平均
        temp_index = np.arange(65)
        preds = np.sum(preds*temp_index,axis=1)
        labels = np.argmax(labels,1)
        mae = mean_absolute_error(labels, preds)
        r_score = r2_score(labels,preds)
        mse = mean_squared_error(labels,preds)

        print(
            f'epoch: {epoch} phase: {phase} loss: {total_loss:.3f} mae: {mae:.3f} mse: {mse:.3f} r_score{r_score:.3f}')

        
        result_list = [self.c['model_name'],self.c['lr'],self.c['seed'],epoch,phase,total_loss,mae]
        
        with open(self.log_path + "/log.csv",'a') as f:
            writer = csv.writer(f)
            writer.writerow(result_list)


        return mae,total_loss,r_score,mse

def tbx_write(tbw,epoch,phase,mae):
    tbw.add_scalar('Mean_Mae/{}'.format(phase),mae,epoch)

def tbx_write(tbw,epoch,phase,mae,loss):
    tbw.add_scalar('Mean_Loss/{}'.format(phase),loss,epoch)
    tbw.add_scalar('Mean_Mae/{}'.format(phase),mae,epoch)
    #self.tb_writer.add_scalar('Recall/{}'.format(phase),self.recall[phase],self.c['n_epoch'])
    #self.tb_writer.add_scalar('Precision/{}'.format(phase),self.precision[phase],self.c['n_epoch'])