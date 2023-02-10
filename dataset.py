import numpy as np
import glob, os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from utils import *
import random
import copy

class JPEG_Dataset_train_gray(Dataset):
    '''
    Dataset for deJPEG train
    '''
    def __init__(self, args):
        self.args = args
        self.image_paths = list()
        for dataset in args.train_dataset:
            self.image_paths.extend(sorted(glob.glob(f'./dataset/{dataset}/*')))
        self.totensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_paths)       

    def __getitem__(self, i):        
        image = cv2.imread(self.image_paths[i], cv2.IMREAD_COLOR) # BGR
        h, w, _ = image.shape
        
        if self.args.double_aug:
            rnd_h = random.randint(0, max(0, h - self.args.patch_size - 8))
            rnd_w = random.randint(0, max(0, w - self.args.patch_size - 8))
            patch = image[rnd_h:rnd_h+self.args.patch_size + 8, rnd_w:rnd_w+self.args.patch_size + 8,:]
        else:
            rnd_h = random.randint(0, max(0, h - self.args.patch_size))
            rnd_w = random.randint(0, max(0, w - self.args.patch_size))
            patch = image[rnd_h:rnd_h+self.args.patch_size, rnd_w:rnd_w+self.args.patch_size,:]

        # augmentation, random flip, rotate
        aug_type = np.random.randint(8)
        patch = augmentation(patch, aug_type)
        if random.random() > 0.25:
            patch = bgr2ycbcr(patch, y_only=True)
        else:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        label = patch.copy()
        
        if random.random() > 0.75:
            qf = np.random.randint(5,85)
        else:
            qf = random.choice([10,20,30,40,50,60,70,80])

        result, encimg = cv2.imencode('.jpg', patch, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
        patch = cv2.imdecode(encimg, 0)


        if self.args.double_aug:
            H, W = patch.shape[:2]
            if random.random() > 0.5:
                rnd_h = random.randint(0, max(0, H - self.args.patch_size))
                rnd_w = random.randint(0, max(0, W - self.args.patch_size))
            else:
                rnd_h = 0
                rnd_w = 0

            patch = patch[rnd_h:rnd_h + self.args.patch_size, rnd_w:rnd_w + self.args.patch_size]
            label = label[rnd_h:rnd_h + self.args.patch_size, rnd_w:rnd_w + self.args.patch_size]
            
            if random.random() > 0.75:
                qf = np.random.randint(5,85)
            else:
                qf = random.choice([10,20,30,40,50,60,70,80])

            result, encimg = cv2.imencode('.jpg', patch, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
            patch = cv2.imdecode(encimg, 0)

        patch = np.reshape(patch, (*patch.shape, 1));
        label = np.reshape(label, (*label.shape, 1));
        
        patch = self.totensor(patch).float()
        label = self.totensor(label).float()
        sigma = torch.ones_like(label) * (100 - qf) / 100
        
        return patch, label, sigma
    

class JPEG_Dataset_train_color(Dataset):
    '''
    Dataset for deJPEG train
    '''
    def __init__(self, args):        
        self.args = args
        self.image_paths = list()
        for dataset in args.train_dataset:
            self.image_paths.extend(sorted(glob.glob(f'./dataset/{dataset}/*')))
        self.totensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_paths)       

    def __getitem__(self, i):
        
        image = cv2.imread(self.image_paths[i], cv2.IMREAD_COLOR) # BGR
        h, w, _ = image.shape

        rnd_h = random.randint(0, max(0, h - self.args.patch_size - 8))
        rnd_w = random.randint(0, max(0, w - self.args.patch_size - 8))
        patch = image[rnd_h:rnd_h+self.args.patch_size, rnd_w:rnd_w+self.args.patch_size,:]

        aug_type = np.random.randint(8)
        patch = augmentation(patch, aug_type)                 
            
        label = copy.deepcopy(patch)
        if random.random() > 0.75:
            qf = np.random.randint(5,85)
        else:
            qf = random.choice([10,20,30,40,50,60,70,80])

        result, encimg = cv2.imencode('.jpg', patch, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
        patch = cv2.imdecode(encimg, 1)
        
        patch = self.totensor(patch).float()
        label = self.totensor(label).float()
        sigma = torch.ones_like(label) * (100 - qf) / 100
        
        return patch, label, sigma


class JPEG_Dataset_val(Dataset):
    '''
    Dataset for deJPEG validation
    '''
    def __init__(self, args):
        self.args = args
        self.image_paths = sorted(glob.glob(f'./dataset/{args.val_dataset}/*'))
        self.totensor = transforms.ToTensor()
        self.qf = args.val_qf
        
    def __len__(self):
        return len(self.image_paths)       

    def __getitem__(self, i):    
        
        if self.args.mode == 'color':
            label = cv2.imread(self.image_paths[i], cv2.IMREAD_COLOR)
            image = copy.deepcopy(label)

            result, encimg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.qf])
            image = cv2.imdecode(encimg, 1)
               
        else:
            label = cv2.imread(self.image_paths[i], cv2.IMREAD_ANYCOLOR)
            if len(label.shape) == 3 and label.shape[-1]==3:
                label = bgr2ycbcr(label, y_only=True)
            image = copy.deepcopy(label)

            result, encimg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.qf])
            image = cv2.imdecode(encimg, 0)
            
        
        if self.args.mode == 'gray':
            image = np.reshape(image, (*image.shape, 1))
            label = np.reshape(label, (*label.shape, 1)) 
        
        image = self.totensor(image).float()
        label = self.totensor(label).float()
        sigma = (100 - int(self.qf)) * torch.ones_like(label) / 100
                
        return image, label, sigma
 
class JPEG_Dataset_val_color_double(Dataset):
    '''
    Dataset for jpeg noise dataset(test)
    '''
    def __init__(self, args):
        self.args = args
        self.image_paths = sorted(glob.glob(f'{self.args.val_dir}/GT/*'))
        self.totensor = transforms.ToTensor()
        self.qf1 = args.qf1
        self.qf2 = args.qf2
        self.qf = args.val_qf
        
    def __len__(self):
        return len(self.image_paths)       

    def __getitem__(self, i):
        filename, ext = os.path.splitext(os.path.basename(self.image_paths[i]))       
        
        label = cv2.imread(f'{self.args.val_dir}/GT/{filename}.png', cv2.IMREAD_COLOR)
        image = copy.deepcopy(label)

        # bgr로 압축
        result, encimg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.qf1])
        image = cv2.imdecode(encimg, 1)
        
        result, encimg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.qf2])
        image = cv2.imdecode(encimg, 1)
        
        
        image = self.totensor(image).float()
        label = self.totensor(label).float()
        sigma = (100 - int(self.qf)) * torch.ones_like(label) / 100
                
        return image, label, sigma, filename