import os,sys,time
import glob
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import cv2

class SDNET(torch.utils.data.Dataset):
    def __init__(self, data_dir,model_name,augmentation=None,blur=False,gaussian=False):
        self.data_dir = data_dir
        #ラベル付のための処理
        if self.data_dir[-1]!='/':
            self.data_dir.append('/')
        self.model_n = model_name
        self.blur = blur
        self.gaussian = gaussian

        self.P_CP = glob.glob(os.path.join(data_dir,'P/CP/*.jpg'))
        self.P_UP = glob.glob(os.path.join(data_dir,'P/UP/*.jpg'))
        self.D_CD = glob.glob(os.path.join(data_dir,'D/CD/*.jpg'))
        self.D_UD = glob.glob(os.path.join(data_dir,'D/UD/*.jpg'))
        self.W_CW = glob.glob(os.path.join(data_dir,'W/CW/*.jpg'))
        self.W_UW = glob.glob(os.path.join(data_dir,'W/UW/*.jpg'))

        self.image_paths = list(itertools.chain.from_iterable([self.P_CP,self.P_UP,self.D_CD,self.D_UD,self.W_CW,self.W_UW]))
        random.shuffle(self.image_paths)
        print(f'data num:{len(self.image_paths)}')
        self.to_tensor = transforms.ToTensor()

        self.augmentation = augmentation

        #print(self.image_paths[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self._load_image(self.image_paths[index],self.model_n)
        
        image_path = self.image_paths[index].replace(self.data_dir,'')
        if image_path[2]=='U':
            label = 0
        elif image_path[2]=='C':
            label = 1
        else:
            print('path error!')
                                                                  
        if self.augmentation is not None:
            augmented = self.augmentation(image=image)
            image = augmented['image']

        image = self.to_tensor(image)
        #label = torch.Tensor([label])                                          

        sample = {'x': image, 'y': label}
        return sample

    def _load_image(self, path,model_n):
        image = cv2.imread(path, 1).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        if model_n == 'resnet101':
            image =cv2.resize(image,(224,224))
            if self.blur:
                n = random.randrange(5,16)
                image = cv2.blur(image,(n,n))
            if self.gaussian:
                noise= np.random.normal(0,50,(224,224,3))
                image = image+noise
        else:
            if self.blur:
                n = random.randrange(5,16)
                image = cv2.blur(image,(n,n))
            if self.gaussian:
                noise= np.random.normal(0,50,(256,256,3))
                image = image+noise
        #[0,1]に変換して出力
        return image/255

class PAIR_SDNET(torch.utils.data.Dataset):
    def __init__(self, data_dir,model_name,blur=False,gaussian=False):
        self.data_dir = data_dir
        #ラベル付のための処理
        if self.data_dir[-1]!='/':
            self.data_dir.append('/')
        self.model_n = model_name
        self.blur = blur
        self.gaussian = gaussian

        self.P_CP = glob.glob(os.path.join(data_dir,'P/CP/*.jpg'))
        self.P_UP = glob.glob(os.path.join(data_dir,'P/UP/*.jpg'))
        self.D_CD = glob.glob(os.path.join(data_dir,'D/CD/*.jpg'))
        self.D_UD = glob.glob(os.path.join(data_dir,'D/UD/*.jpg'))
        self.W_CW = glob.glob(os.path.join(data_dir,'W/CW/*.jpg'))
        self.W_UW = glob.glob(os.path.join(data_dir,'W/UW/*.jpg'))

        self.image_paths = list(itertools.chain.from_iterable([self.P_CP,self.P_UP,self.D_CD,self.D_UD,self.W_CW,self.W_UW]))
        random.shuffle(self.image_paths)
        self.image_paths2 = random.sample(self.image_paths,len(self.image_paths))
        print(f'data num:{len(self.image_paths)}')
        self.to_tensor = transforms.ToTensor()

        #print(self.image_paths[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        blur,image = self._load_image(self.image_paths[index],self.image_paths2[index],self.model_n)
        
        image_path = self.image_paths[index].replace(self.data_dir,'')
        if image_path[2]=='U':
            label = 0
        elif image_path[2]=='C':
            label = 1
        else:
            print('path error!')
                                                            
        image = self.to_tensor(image)   
        blur = self.to_tensor(blur)                                     

        sample = {'x': blur, 'y': label,'gt': image}
        return sample

    def _load_image(self, path1,path2,model_n):
        #ノイズ画像
        image = cv2.imread(path1, 1).astype('uint8')
        #綺麗な画像
        image2 = cv2.imread(path2, 1).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        if model_n == 'resnet101':
            image =cv2.resize(image,(224,224))
            image2 =cv2.resize(image2,(224,224))
            if self.blur:
                n = random.randrange(5,16)
                image = cv2.blur(image,(n,n))
            if self.gaussian:
                noise= np.random.normal(0,50,(224,224,3))
                image = image+noise
        else:
            if self.blur:
                n = random.randrange(5,16)
                image = cv2.blur(image,(n,n))
            if self.gaussian:
                noise= np.random.normal(0,50,(256,256,3))
                image = image+noise
        #[0,1]に変換して出力
        return image/255,image2/255