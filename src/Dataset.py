import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config
from PIL import Image
import os
import numpy as np
from utils import isint


class NuclearCataractDatasetBase(Dataset):
    def __init__(self, root, image_list_file, transform=None):
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(',')
                if isint(items[2]):
                    label = self.get_label(int(items[6][0]))
                    label[0] = 1 if (label[0]>=2) else 0
                    for i in range(16):
                        image_name = items[2] + '_' + items[3] + '_' + '{:0=3}'.format(int(i)) + '.jpg'
                        image_name = os.path.join(root,image_name)
                        image_names.append(image_name)
                        labels.append(label[0])

        self.image_names = np.array(image_names)
        self.labels = np.array(labels)
        self.transform = transform
        '''self.label_names = ['Nondisease',
                            'Mild',
                            'Moderateness',
                            'Severe'] '''

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('L')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        if label == 1:
            label = 0
        else:
            label = 1
        return image,torch.Tensor([label])

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
        if label_base == 1:
            return [1]
        else:
            return [0]

def load_dataloader(batch_size):
    train_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.CenterCrop(config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, ),
                                                 (0.5, ))])
    test_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.CenterCrop(config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, ),
                                                 (0.5, ))])
    dataset = {}
    dataset['train'] = \
        NuclearCataractDataset(root=config.data_root,
                                  image_list_file=config.train_info_list,
                                  transform=train_transform)
    dataset['test'] = \
        NuclearCataractDataset(root=config.data_root,
                                  image_list_file=config.test_info_list,
    
                                  transform=test_transform)

    return dataset
    '''
    CV実装のため、データセットのみの実装
    dataloader = {}
    
    dataloader['train'] = \
        torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=0)
    dataloader['test'] = \
        torch.utils.data.DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    num_workers=0)
    return dataloader
    '''

