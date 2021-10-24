import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os
import config
from PIL import Image
from glob import glob

def sigmoid(x):
    return 1/(1+np.exp(-x))

def iterate(d, param={}):
    d, param = d.copy(), param.copy()
    d_list = []

    for k, v in d.items():
        if isinstance(v, list):
            for vi in v:
                d[k], param[k] = vi, vi
                d_list += iterate(d, param)
            return d_list

        if isinstance(v, dict):
            add_d_list = iterate(v, param)
            if len(add_d_list) > 1:
                for vi, pi in add_d_list:
                    d[k] = vi
                    d_list += iterate(d, pi)
                return d_list

    return [[d, param]]

def isint(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True

def init_weights(m):
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data,a=math.sqrt(5))
        if m.bias is not None:
            fan_in,_ = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias.data,-bound,bound)
            
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)

def one_hot_encoding(n):
    if isint(n):
        one_hot = [0]*config.n_classification
        one_hot[n] = 1
        return one_hot
    else:
        print('Please use an integer of n.')

def calc_normal_distribution(x, mu, sigma=1):
    return 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(x - mu)**2 / (2 * (sigma**2)))

#nを中心とした正規分布を返す。
def normal_distribution(n,sigma=1):
    nd = [calc_normal_distribution(x,n) for x in range(config.n_classification)]
    return nd

#ビニング処理。年齢をn分割して、labelを該当場所に割り振る。
#bincenterは割り振られた場所の真ん中の値
def execute_bining(n,label):
    base_num = np.arange(65)+21
    base_num[0] = 15
    base_num[-1] = 91
    
    bin_list,bins = pd.cut(base_num,bins=n,retbins=True,labels=False)
    
    n_class = bin_list[label-21] if label>21 else 0
    bincenter = (bins[n_class]+bins[n_class+1])/2

    
    return n_class,bincenter

def ret_bins(n):
    base_num = np.arange(65)+21
    base_num[0] = 15
    base_num[-1] = 91
    
    _,bins = pd.cut(base_num,bins=n,retbins=True,labels=False)

    return bins

