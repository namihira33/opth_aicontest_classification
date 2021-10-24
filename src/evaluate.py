import os
import sys
import time
from datetime import datetime
import random

from tqdm import tqdm
import numpy as np
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression

import config
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import tensorboardX as tbx

from torch.utils.data import DataLoader 
from network import *
from Dataset import load_dataloader

import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

import pytab as pt
import seaborn as sns

c = {
    'model_name': 'Resnet34',
    'seed': [0], 'bs': 64, 'lr': [1e-4], 'n_epoch': [10]
}

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

#グラフ内で日本語を使用可能にする。
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Evaluater():
    def __init__(self,c):
        self.dataloaders = {}
        self.c = c
        self.cnt = 0
        now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        model_path = ''
        args = len(sys.argv)
        with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv')) as f:
             lines = [s.strip() for s in f.readlines()]
        if args < 2 :
             target_data = lines[-1].split(',')
        else:
             if int(sys.argv[1])<=0:
                 print('Use the first data')
                 target_data = lines[-1].split(',')
             else:
                 try:
                     target_data = lines[int(sys.argv[1])].split(',')
                 except IndexError:
                     print('It does not exit. Use the first data')
                     target_data = lines[-1].split(',')

        self.n_ex = '{:0=2}'.format(int(target_data[1]))
        self.c['model_name'] = target_data[2]
        self.c['n_epoch'] = '{:0=3}'.format(int(target_data[3]))
        temp = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep.pth'
        model_path = os.path.join(config.MODEL_DIR_PATH,temp)


        self.net = make_model(self.c['model_name']).to(device)
        self.net.load_state_dict(torch.load(model_path,map_location=device))

        #モデル構造を可視化
        dummy_input = torch.rand(64,3,224,224)
        with tbx.SummaryWriter() as w:
            w.add_graph(self.net,(dummy_input.to(device)))

    def run(self):
            self.dataset = load_dataloader(0.25)
            test_dataset = self.dataset['test']
            self.dataloaders['test'] = DataLoader(test_dataset,self.c['bs'],
                    shuffle=True,num_workers=os.cpu_count())

            preds, labels,total_loss,accuracy= [], [],0,0
            right,notright = 0,0
            self.net.eval()


            #fig,axes = plt.subplots(4,8,figsize=(16,16))
            #fig,axes = plt.subplots(2,4,figsize=(16,8))
            #plt.subplots_adjust(wspace=0.4, hspace=0.6)

            for inputs_,labels_,paths_,_ in tqdm(self.dataloaders['test']):
                inputs_ = inputs_.to(device)
                labels_ = labels_.to(device)

                torch.set_grad_enabled(False)
                outputs_ = self.net(inputs_)
                #loss = self.criterion(outputs_, labels_)
                #total_loss += loss.item()

                #画像・ラベル・推定値を表示させてみる。
                #表示させられるのだけど、標準化されているので元の形を保っていない。
                #これを出す方法。
                #conf = 0.2
                paths_ = np.array(paths_)
                #predicts_percent = outputs_.detach().cpu().numpy()
                #predicts_percent = np.max(predicts_percent,axis=1)
                predicts = np.argmax(outputs_.detach().cpu().numpy(),axis=1)
                #predicts = predicts[predicts_percent<conf]
                #answers = np.argmax(labels_.detach().cpu().numpy(),axis=1)
                #temp = abs(predicts-answers)>20
                #predicts = predicts[temp]
                #answers = answers[temp]
                #answers = answers[predicts_percent<conf]
                #paths_ = paths_[predicts_percent<conf]
                #paths_ = paths_[temp]
                #predicts_percent = predicts_percent[predicts_percent<conf]
                #predicts_percent = predicts_percent[temp]
                #root = config.data_root

                #for i,(p,pp,pd,an) in enumerate(zip(paths_,predicts_percent,predicts,answers)):
                #    image_name = os.path.join(root,p)
                #    im = Image.open(image_name)
                #    title1 = '予測:' + str(pd+21) + '歳 ' + '答え:' + str(an+21) + '歳'
                #    title2 = '確信度 : ' + '{:.2f}'.format(pp)
                #    axes[(self.cnt%8)//4][self.cnt%4].imshow(im)
                #    axes[(self.cnt%8)//4][self.cnt%4].set_title(title1)
                #    axes[(self.cnt%8)//4][self.cnt%4].title.set_size(18)
                #    axes[(self.cnt%8)//4][self.cnt%4+1].set_ylim(0,1)
                #    axes[(self.cnt%8)//4][self.cnt%4+1].set_xlim(0,1)
                #    axes[(self.cnt%8)//4][self.cnt%4+1].set_aspect('equal', adjustable='box')
                #    axes[(self.cnt%8)//4][self.cnt%4+1].set_title(title2)
                #    axes[(self.cnt%8)//4][self.cnt%4+1].title.set_size(18)
                #    axes[(self.cnt%8)//4][self.cnt%4+1].bar(0.5,pp,width=0.4,align='center',tick_label='confidence')
                #    self.cnt += 2
                #    if not (self.cnt%8):
                #        save_path = './log/images/experiment' + str(self.cnt//8) + '.png'
                #        fig.savefig(save_path)
                #        fig,axes = plt.subplots(2,4,figsize=(16,8))



                #fig,ax = plt.subplots()
                #plt.xlim(-1,1)
                #ax.set_ylim(0,1)
                #hist = ax.bar(0,np.max(predicts_percent[0]),width=0.4,align='center',tick_label='confidence')
                #print(np.max(predicts_percent[0]))
                #plt.plot()
                #fig.savefig('./log/images/experiment_bar.png')

                #predicts = np.argmax(outputs_.detach().cpu().numpy(),axis=1)
                #answers = np.argmax(labels_.detach().cpu().numpy(),axis=1)



                #root = config.data_root
                #image_name = os.path.join(root,path)
                #im = Image.open(image_name)

                #fig,ax = plt.subplots()
                #ax.imshow(im)
                #title = '予測:' + str(temp1[0]+21) + '歳  答え:' + str(temp2[0]+21) + '歳'
                #ax.set_title(title)
                #fig.savefig('./log/images/experiment.png')
                


                preds += [outputs_.detach().cpu().numpy()]
                labels += [labels_.detach().cpu().numpy()]
                #total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

            #if self.cnt%8:
            #    while (self.cnt%8):
            #        axes[(self.cnt%8)//4][self.cnt%4].axis('off')
            #        axes[(self.cnt%8)//4][self.cnt%4+1].axis('off')
            #        self.cnt += 2

            #    save_path = './log/images/experiment' + str(self.cnt//8) + '_last.png'
            #    fig.savefig(save_path)

            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            #total_loss /= len(preds)

            #worst_id = np.argmax(preds-labels)
            #worst = (preds-labels).max()

            #a = np.max(preds,1)
            #preds = np.argmax(preds[a>0.9],axis=1)
            #labels = np.argmax(labels[a>0.9],axis=1)
            #fig,ax = plt.subplots()
            #p = ax.hist(a)
            #fig_path = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep_hist.png'
            #fig.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))
            
            #preds = np.argmax(preds,1)
            temp_index = np.arange(config.n_classification)
            preds = np.sum(preds*temp_index,axis=1)            
            labels = np.argmax(labels,1)
            preds += 21
            labels += 21

            sns.set()
            sns.set_style('whitegrid')
            sns.set_palette('Set3')
            fig,ax = plt.subplots(figsize=(16,8))
            ax.set_xlabel('Predict Age')
            ax.set_ylabel('Num of Datas')
            ax.set_title('Predict Age Histgram')
            hist = ax.hist(preds,bins=65)
            fig.savefig('./log/images/pred_Regression_hist.png')

            #threshold = 1.01
            #right += ((preds-labels) < threshold).sum()
            #notright += len(preds) - ((preds - labels) < threshold).sum()

            #print(right,len(preds))

            #accuracy = right / len(preds)
            mae = mean_absolute_error(preds,labels)
            r_score = r2_score(preds,labels)

            #print('accuracy :',accuracy)
            #print('MAE :',mae)
            #print('AE : ',mae*len(preds))

            #評価結果を図示する。
            data = {
            #'Accuracy' : ['{:.2f}'.format(accuracy)],
            'R2-score' : ['{:.2f}'.format(r_score)],
            'Age MAE' : ['{:.2f}'.format(mae)]
            }
            tb = pt.table(data=data,th_type='dark')
            fig = tb.figure
            fig.suptitle('Score')
            fig.set_figheight(2)
            fig.set_figwidth(6)
            fig_path = 'scoretable.png'
            fig.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))



            #lr = LinearRegression()
            #preds = preds.reshape(-1,1)
            #lr.fit(preds.reshape(-1,1),labels)
            #fig,ax = plt.subplots()
            #ax.scatter(preds,labels)
            #ax.plot(preds,lr.predict(preds),color='red')
            #fig_path = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep_regression.png'
            #plt.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))


            #fig,ax = plt.subplots()
            #ax.bar(['Acc','Mae','R-score'],[accuracy,mae,r_score],width=0.4,tick_label=['Accuracy','Mae','R-Score'],align='center')
            #ax.grid(True)
            #fig_path = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep_graph.png'
            #fig.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))



if __name__ == '__main__':
    evaluater = Evaluater(c)
    evaluater.run()
