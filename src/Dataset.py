import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config
from PIL import Image
import os
import numpy as np
from utils import *


class NuclearCataractDatasetBase(Dataset):
    def __init__(self, root, image_list_file, transform=None):
        image_names = []
        labels = []
        indexes = []
        images = []
        self.image_list_file = image_list_file

        with open(image_list_file,"r") as f:
            for line in f:
                items = line.split(',')
                if isint(items[0]):
                    index = int(items[0])
                    if image_list_file == config.contest_test_list:
                        image_name = items[1].rstrip('\n') + '.jpg'
                    else:
                        image_name = items[1]
                        label = self.get_label(int(items[2]))
                        labels.append(label[0])

                    image_name = os.path.join(root,image_name)
                    indexes.append(index)
                    image_names.append(image_name)



        self.mode = 'train' if ((image_list_file == config.train_info_list) or (image_list_file == config.contest_train_list))  else 'test'
        self.image_names = np.array(image_names)
        self.labels = np.array(labels)
        self.indexes = np.array(indexes)
        self.transform = transform
        #self.images = images
 
    #trainの時は画像とラベル、testのときはそれに加えて画像パスも返すようにできない？
    def __getitem__(self, index):
        image_name = self.image_names[index]
        #image = self.images[index]

        if self.image_list_file == config.contest_test_list:
            label = 0
        else : 
            label = self.labels[index]
        item_index = self.indexes[index]

        #DataAugmentationを使うときはこうする。
        image = Image.open(image_name).convert('RGB')
        #左目だったら、右目に回転
        image = transforms.RandomHorizontalFlip(p=1.0)(image) if image_name[-5] == 'L' else image
        if self.transform is not None:
            image = self.transform(image)
            
        #label = one_hot_encoding(label)
        label = normal_distribution(label)

        return (image,torch.Tensor(label),item_index) if (self.mode == 'train') else (image,torch.Tensor(label),image_name,item_index)


    def __len__(self):
        return len(self.image_names)
        #return 1000

    def get_label(self, label_base):
        pass

'''
class NuclearCataractDatasetBinary(NuclearCataractDatasetBase):
    def get_label(self, label_base):
        if sum([int(i) for i in label_base]) != 0:
            # with disease
            return [1]
        else:
            # without disease
            return [0]
            '''
        

class NuclearCataractDataset(NuclearCataractDatasetBase):
    def get_label(self, label_base):
        if label_base < 22:
            label_base = 0
        elif label_base > 84:
            label_base = 64
        else:
            label_base -= 21
        return [label_base]

def load_dataloader(p):
    train_transform = \
        transforms.Compose([#transforms.Resize(config.image_size),
                            #transforms.CenterCrop(config.image_size),
                            transforms.RandomResizedCrop(config.image_size,scale=(p,1.0),ratio=(1,1)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],
                                                 [0.229,0.224,0.225])
                            ])

    valid_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.CenterCrop(config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],
                                                 [0.229,0.224,0.225])])

    test_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.CenterCrop(config.image_size),
                            transforms.ToTensor(),
                            #])#,
                            transforms.Normalize([0.485,0.456,0.406],
                                                 [0.229,0.224,0.225])])
    dataset = {}
    dataset['train'] = \
        NuclearCataractDataset(root=config.contest_train,
                                  image_list_file=config.train_info_list,
                                  transform=train_transform)

    dataset['valid'] = \
        NuclearCataractDataset(root=config.contest_train,
                                  image_list_file=config.train_info_list,
                                  transform=valid_transform)
    
    dataset['test'] = \
        NuclearCataractDataset(root=config.contest_train,
                                  image_list_file=config.test_info_list,
                                  transform=test_transform)

    dataset['contest_train'] = \
        NuclearCataractDataset(root=config.contest_train,
                                  image_list_file=config.contest_train_list,
                                  transform=train_transform)
    dataset['contest_valid'] = \
        NuclearCataractDataset(root=config.contest_train,
                                  image_list_file=config.contest_train_list,
                                  transform=valid_transform)      

    dataset['contest_test'] = \
        NuclearCataractDataset(root=config.contest_test,
                                image_list_file=config.contest_test_list,
                                transform=test_transform)

    return dataset

