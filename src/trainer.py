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
import pandas as pd


torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

class Trainer():
    def __init__(self, c):
        self.dataloaders = {}
        self.prms = []
        self.search = c
        self.n_seeds = len(c['seed'])
        self.n_splits = 5        

        self.now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        self.log_path = os.path.join(config.LOG_DIR_PATH,
                                str(self.now))
        os.makedirs(self.log_path, exist_ok=True)

        with open(self.log_path + "/log.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerow(['-'*20 + 'Log File' + '-'*20])

        

    def run(self):
        #実行時間計測とmae代入準備
        start = time.time()

        #ヒートマップ描画用のリスト
        validheat,heat_index = [],[]

        #Seed平均を取るためのリスト
        seed_valid = []

        #valid予測値出力用のリスト
        ensemble_list = []
        test_preds = []

        #CSVファイルヘッダー記述
        with open(self.log_path + "/log.csv",'a') as f:
            writer = csv.writer(f)
            writer.writerow(['model_name','lr','seed','epoch','phase','total_loss','mae'])

        for n_iter,(c,param) in enumerate(iterate(self.search)):
            print('Parameter :',c)
            self.c = c
            random.seed(self.c['seed'])
            torch.manual_seed(self.c['seed'])

            mn = self.c['model_name']
            self.net = make_model(mn).to(device)
            self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
            #self.criterion = nn.BCELoss()
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.net = nn.DataParallel(self.net)

            memory = {}
            memory2 = {}
            for phase in ['learning','valid']:
                if self.c['cv'] == 0:
                    memory[phase] = [[] for x in range(1)]  #self.n_splits
                    memory2[phase] = [[] for x in range(1)] #self.n_splits
                else:
                    memory[phase] = [[] for x in range(self.n_splits)]
                    memory2[phase] = [[] for x in range(self.n_splits)]

            

            self.dataset = load_dataloader(self.c['p'])
            kf = KFold(n_splits=5,shuffle=True,random_state=self.c['seed'])

            for a,(learning_index,valid_index) in enumerate(kf.split(self.dataset['contest_train'])):
                #データセットが切り替わるたびに、ネットワークの重み,バイアスを初期化
                self.net.apply(init_weights)
                self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
                    
                learning_dataset = Subset(self.dataset['contest_train'],learning_index)
                self.dataloaders['learning'] = DataLoader(learning_dataset,self.c['bs'],
                shuffle=True,num_workers=os.cpu_count())
                
                valid_dataset = Subset(self.dataset['contest_valid'],valid_index)
                self.dataloaders['valid'] = DataLoader(valid_dataset,self.c['bs'],
                shuffle=True,num_workers=os.cpu_count())

                self.tb_writer = tbx.SummaryWriter()
                self.earlystopping = EarlyStopping(patience=15,verbose=True,delta=0)
                for epoch in range(1, self.c['n_epoch']+1):

                    learningmae,learningloss,learningr_score \
                        = self.execute_epoch(epoch, 'learning')
                    self.tb_writer.add_scalar('Loss/{}'.format('learning'),learningloss,epoch)
                    self.tb_writer.add_scalar('Mae/{}'.format('learning'),learningmae,epoch)

                    
                    if not self.c['evaluate']:
                        validmae,validloss,validr_score,valid_preds,valid_labels,valid_indexes\
                            = self.execute_epoch(epoch, 'valid')
                        self.tb_writer.add_scalar('Loss/{}'.format('valid'),validloss,epoch)
                        self.tb_writer.add_scalar('Mae/{}'.format('valid'),validmae,epoch)

                        memory['valid'][a].append(validmae)
                        memory2['valid'][a].append(validloss)

                        if (230<epoch):
                            self.earlystopping(validmae,self.net,epoch)
                            if self.earlystopping.early_stop:
                                print("Early Stopping")
                                print('Stop epoch : ',epoch)
                                break

                        temp = validmae,epoch,self.c
                        self.prms.append(temp)          
                
                #n_epoch後の処理                
                self.tb_writer.close()

                #アンサンブル用 予測した値をindexに紐づけて保存
                self.net.load_state_dict(torch.load(self.earlystopping.path))
                validmae,_,_,valid_preds,valid_labels,valid_indexes\
                    = self.execute_epoch(self.earlystopping.epoch,'valid')
                
                #最高Valid値を記録。
                with open("./log" + "/early_log.csv",'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(['-'*20 + 'early stopping ' + '-'*20])
                    writer.writerow(['divide:',a,'stop epoch',epoch,'min_mae',validmae])

                for a,b,c in zip(valid_indexes,valid_preds,valid_labels):
                    ensemble_list.append((a,b+21,c+21))

                #各分割での学習モデルを使って、テストデータに対する予測を出す。
                self.dataset = load_dataloader(self.c['p'])
                test_dataset = self.dataset['contest_test']
                self.dataloaders['test'] = DataLoader(test_dataset,self.c['bs'],
                    shuffle=False,num_workers=os.cpu_count())

                preds, labels,paths,test_indexes,total_loss,accuracy= [],[],[],[],0,0
                self.net.eval()

                for inputs_,labels_,paths_,indexes_ in tqdm(self.dataloaders['test']):
                    inputs_ = inputs_.to(device)
                    labels_ = labels_.to(device)

                    torch.set_grad_enabled(False)
                    outputs_ = self.net(inputs_)
                    loss = self.criterion(outputs_, labels_)

                    preds += [outputs_.detach().cpu().numpy()]
                    labels += [labels_.detach().cpu().numpy()]
                    test_indexes += [indexes_.detach().cpu().numpy()]
                    paths  += paths_

                preds = np.concatenate(preds)
                preds = np.argmax(preds,axis=1)
                labels = np.concatenate(labels)
                test_indexes = np.concatenate(test_indexes)
                for t_id,pd in zip(test_indexes,preds):
                    test_preds.append((t_id,pd+21))


                if not self.c['cv']:
                    break

            #分割交差検証後の処理
            #memory['learning'] = list(np.mean(memory['learning'],axis=0))
            #memory2['learning'] = list(np.mean(memory2['valid'],axis=0))
            #if not self.c['evaluate']:
            #    memory['valid'] = list(np.mean(memory['valid'],axis=0))
            #    memory2['valid'] = list(np.mean(memory2['valid'],axis=0))
            #    seed_valid.append(memory['valid'])

            #平均をTensorboardに記録。
            #if self.c['cv']:
            #    self.tb_writer = tbx.SummaryWriter()
            #    for phase in ['learning','valid']:
            #        for b in range(len(memory[phase])):
            #            tbx_write(self.tb_writer,b+1,phase,memory[phase][b],memory2[phase][b])
            #            with open(self.log_path + "/log.csv",'a') as f:
            #                writer = csv.writer(f)
            #                writer.writerow([self.c,phase,'ValidMAE',memory[phase][b]])
            #        for b in range(len(memory[phase])):
            #            with open(self.log_path + "/log.csv",'a') as f:
            #                writer = csv.writer(f)
            #                writer.writerow([self.c,phase,'ValidLoss',memory2[phase][b]])
            #    #JSON形式でTensorboardに保存した値を残しておく。
            #    self.tb_writer.export_scalars_to_json('./log/all_scalars.json')
            #    self.tb_writer.close()

            #乱数シードiterごとに、平均を取り、これを記録。
            if not ((n_iter+1)%self.n_seeds):
                temp = self.n_seeds * self.n_splits
                #print(memory['valid'])
                #seed_valid = np.mean(seed_valid,axis=0)
                #validheat.append(memory['valid'])
                #heat_index.append(self.c['lr'])
                #seed_valid = []

        #パラメータiter後の処理。

        best_prms = sorted(self.prms,key=lambda x:x[0])
        with open(self.log_path + "/log.csv",'a') as f:
            writer = csv.writer(f)
            writer.writerow(['-'*20 + 'bestparameter' + '-'*20])
            writer.writerow(['model_name','lr','seed','n_epoch','mae'])
            writer.writerow(best_prms[0:10])
            ensemble_list = sorted(ensemble_list,key=lambda x:x[0])
            test_preds = sorted(test_preds,key=lambda x:x[0])

#       print(ensemble_list[0:10])
#       print(np.mean(test_preds,axis=0)[:10])

        #学習率・10epoch経過後のヒートマップの描画
        #    validheat = [l[::5] for l in validheat[:]]
        #    print(validheat)
        #    fig,ax = plt.subplots(figsize=(16,8))
        #    xtick = list(map(lambda x:5*x-4,list(range(1,len(validheat[0])+1))))
        #    xtick = [str(x) + 'ep' for x in xtick]
        #    sns.heatmap(validheat,annot=True,cmap='Set3',fmt='.2f',
        #        xticklabels=xtick,yticklabels=heat_index,vmin=2.5,vmax=10,
        #        cbar_kws = dict(label='Valid Age MAE'))
        #    ax.set_ylabel('learning rate')
        #    ax.set_xlabel('num of epoch')
        #    ax.set_title('num of seeds : ' + str(self.n_seeds))
        #    fig.savefig('./log/images/'+self.now + 'train_ep.png')

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

        #モデルの情報に基づいて、今回のValidでの予測値・テストデータに対する予測値を保存する。
        #test_predsをn回ごとに平均を取る。
        list_t_pd,s = [],0
        for i,t_pd in enumerate(test_preds):
            s += t_pd[1]
            if (i%5==4):
                s/=5
                list_t_pd.append(s)
                s = 0
        print(list_t_pd)
        with open(os.path.join(config.LOG_DIR_PATH,'second_model.csv'),'w') as f:
            writer = csv.writer(f)
            writer.writerow(['-----Model Predict Value------'])
            for p,pd,l in ensemble_list:
                writer.writerow([pd,l])

            for t_pd in list_t_pd:
                writer.writerow([t_pd])

    #1epochごとの処理
    def execute_epoch(self, epoch, phase):
        preds, labels,indexes,ages,total_loss= [],[],[],[],0
        if phase == 'learning':
            self.net.train()
        else:
            self.net.eval()

        for inputs_, labels_, indexes_ in tqdm(self.dataloaders[phase]):
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)
            self.optimizer.zero_grad()

            #preds_ = np.argmax(preds_,axis=1)
            #for i in range(len(preds_)):
            #    pred,_ = execute_bining(15,preds_[i][0]+21)
            #    preds_[i] = pred

            #labels_ = np.argmax(labels_,axis=1)
            #for i in range(len(labels_)):
            #    label,_ = execute_bining(15,labels_[i][0]+21)
            #    ages.append(label)


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
            indexes += [indexes_.detach().cpu().numpy()]
            total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        indexes = np.concatenate(indexes)
        total_loss /= len(preds)
        
        #ハードな答えの出し方
        #preds = np.argmax(preds,1)
        #for i in range(len(preds)):
        #    bins = ret_bins(config.n_classification)
        #    bincenter = (bins[preds[i]+1] + bins[preds[i]])/2
        #    preds[i] = round(bincenter,2)
        
        #ソフトな答えの出し方 加重平均
        temp_index = np.arange(config.n_classification)
        preds = np.sum(preds*temp_index,axis=1)

        #エンコーディングをもとに戻す。
        labels = np.argmax(labels,1)


        mae = mean_absolute_error(labels, preds)
        r_score = r2_score(labels,preds)

        print(
            f'epoch: {epoch} phase: {phase} loss: {total_loss:.3f} mae: {mae:.3f} r_score{r_score:.3f}')

        
        result_list = [self.c['model_name'],self.c['lr'],self.c['seed'],epoch,phase,total_loss,mae]
        
        with open(self.log_path + "/log.csv",'a') as f:
            writer = csv.writer(f)
            writer.writerow(result_list)
        
        return (mae,total_loss,r_score) if (phase=='learning') else (mae,total_loss,r_score,preds,labels,indexes)


def tbx_write(tbw,epoch,phase,mae):
    tbw.add_scalar('Mean_Mae/{}'.format(phase),mae,epoch)

def tbx_write(tbw,epoch,phase,mae,loss):
    tbw.add_scalar('Mean_Loss/{}'.format(phase),loss,epoch)
    tbw.add_scalar('Mean_Mae/{}'.format(phase),mae,epoch)

#early stoppingクラス 拝借
class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, delta=0,path='./model/checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    
        self.verbose = verbose      
        self.counter = 0
        self.epoch = 0            
        self.best_score = None     
        self.early_stop = False     
        self.val_loss_min = np.Inf   
        self.path = path
        self.delta = delta       

    def __call__(self, val_loss, model,epoch):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   
            self.checkpoint(val_loss,model,epoch)  
        elif score < self.best_score + self.delta:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  
            self.epoch = epoch
            self.checkpoint(val_loss, model,epoch)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model,epoch):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation mae decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        torch.save(model.module.state_dict(),'./model/evaluate.pth') #評価用のpathにベストモデルを保存
        self.val_loss_min = val_loss 