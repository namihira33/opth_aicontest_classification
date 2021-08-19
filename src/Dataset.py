import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config
from PIL import Image
import os
import numpy as np


class NuclearCataractDatasetBase(Dataset):
    def __init__(self, root, image_list_file, transform=None):
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(',')
                image_name = '{:0=4}.format(items[0])' + '.jpg'
                label = self.get_label(items[3])
                image_name = os.path.join(root, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = np.array(image_names)
        self.labels = np.array(labels)
        self.transform = transform
        '''self.label_names = ['Atelectasis',
                            'Cardiomegaly',
                            'Effusion',
                            'Infiltration',
                            'Mass',
                            'Nodule',
                            'Pneumonia',
                            'Pneumothorax',
                            'Consolidation',
                            'Edema',
                            'Emphysema',
                            'Fibrosis',
                            'Pleural_Thickening',
                            'Hernia'] '''

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

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
        return [int(i) for i in label_base]

def load_dataloader(batch_size):
    train_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])
    valid_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])
    train_dataset = \
        NuclearCataractDataset(root=config.data_root,
                                  image_list_file=config.train_imfo_list,
                                  transform=train_transform)
    valid_dataset = \
        NuclearCataractDataset(root=config.data_root,
                                  image_list_file=config.valid_imfo_list,
                                  transform=valid_transform)
    dataloader = {}
    
    dataloader['train'] = \
        torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=8)
    dataloader['test'] = \
        torch.utils.data.DataLoader(valid_dataset,
                                    batch_size=batch_size,
                                    num_workers=8)
    return dataloader


'''
def load_dataloader_binary(batch_size):
    train_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])
    valid_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])
    train_dataset = \
        NuclearCataractDatasetBinary(root=config.data_root,
                                        image_list_file=config.train_imfo_list,
                                        transform=train_transform)
    valid_dataset = \
        NuclearCataractDatasetBinary(root=config.data_root,
                                        image_list_file=config.valid_imfo_list,
                                        transform=valid_transform)
    train_dataloader = \
        torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=4)
    valid_dataloader = \
        torch.utils.data.DataLoader(valid_dataset,
                                    batch_size=batch_size,
                                    num_workers=4)
    return train_dataloader, valid_dataloader
    '''

#bs = 4
#dataloader = load_dataloader(bs)

#batch_iterator = iter(dataloader['train'])
#inputs,labels = next(batch_iterator)

#print(inputs.size())
#print(labels)