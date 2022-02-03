import os,sys,time
import glob
import random
from tqdm import tqdm
import argparse
import copy

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader

from sklearn.metrics import confusion_matrix
import cv2
import albumentations as A
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

import dataloader
import models

criterion = nn.BCELoss().to('cuda')
def compute_loss(label, preds):
    return criterion(preds, label)

#Augumentationのための関数                                                      
def get_transforms(flag='train'):
    if flag == 'train':
        #訓練時のAugumentation                                                  
        transforms = [
                      A.HorizontalFlip(p=0.5),
                      A.VerticalFlip(p=0.5)
                      ]
    else:
        #テスト時のtransform                                                    
        transforms = []

    return A.Compose(transforms)

def train_model(train_loader,model_l,optim_l,loss_func,train_size,epoch,device):
    # training                                                                  
    all_loss = list()

    [model,netG,netD] = model_l
    [optimizer,optim_G,optim_D] = optim_l

    model.train()
    netG.eval()
    netD.eval()

    for batch in tqdm(train_loader):
        noise_image, label = batch['x'].float().to(device), batch['y'].to(device)
        bs,w,h,c = noise_image.size()
        n = netG(noise_image)
        x = noise_image+n
        pred = model(x)
        loss = loss_func(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ログ                                                                  
        all_loss.append(bs*loss.item())

    train_loss = sum(all_loss)/train_size
    print()
    print('Train Loss:',train_loss)
    
    return train_loss

def train_GAN(train_loader,model_l,optim_l,loss_func,train_size,epoch,device):
    # training                                                                  
    loss_G_l = list()
    loss_D_l = list()

    [model,netG,netD] = model_l
    [optimizer,optim_G,optim_D] = optim_l

    netG.train()
    netD.train()

    for batch in train_loader:
        noise_image,clean_image = batch['x'].float().to(device),batch['gt'].float().to(device)
        bs,w,h,c = noise_image.size()
        n = netG(noise_image)
        x = noise_image+n

        optim_G.zero_grad()
        optim_D.zero_grad()
        
        #識別器の学習
        preds = netD(clean_image)
        t = torch.ones(bs).float().to(device)
        loss_D_real = compute_loss(t, preds)
        preds = netD(x.detach())
        t = torch.zeros(bs).float().to(device)
        loss_D_fake = compute_loss(t, preds)
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        loss_D_l.append(bs*loss_D.item())  
        optim_D.step() 
        
        #生成器の学習 
        preds = netD(x) # 偽物画像に対する予測
        t = torch.ones(bs).float().to(device) # 偽物画像のラベルを「本物画像」に
        loss_G = compute_loss(t, preds)
        loss_G.backward()
        optim_G.step()
        # ログ                                                                  
        loss_G_l.append(bs*loss_G.item()) 

    loss_G = sum(loss_G_l)/train_size
    loss_D = sum(loss_D_l)/train_size
    print()
    print('Train G-Loss:',loss_G)
    print('Train D-Loss:',loss_D)
    
    return loss_G,loss_D

def valid(valid_loader,model_l,best_score,best_models):
    # validation                                                                
    all_preds = list()
    all_labels = list()

    [model,netG,netD] = model_l
    model.eval()
    netG.eval()
    netD.eval()
    with torch.no_grad():
        for batch in valid_loader:
            # 推論                                                              
            image, label = batch['x'].float().to('cuda'), batch['y'].float().to('cuda')
            pred = model(netG(image)+image)
            # ログ 
            pred = torch.where(pred[:,0]<pred[:,1],1,0)                                                             
            pred = pred.detach().cpu().numpy()
            #print(pred.shape)
            label = label.detach().cpu().numpy()

            [all_preds.append(i) for i in pred.ravel()]
            [all_labels.append(i) for i in label.ravel()]
    # 精度計算 
    cm = confusion_matrix(all_labels, all_preds)   
    #print(cm)
    tn, fp, fn, tp = cm.flatten()
    print('tp:', tp, '  fn:', fn, '  fp:', fp, '  tn:', tn)
    acc = (tp+tn)/(tn+fp+fn+tp)
    precision = tp/(tp+fp+1e-16)
    recall = tp/(tp+fn+1e-16)    
    f1 = 2*precision*recall/(precision+recall+1e-16)
    print('acc:', acc, '  precision:', precision, '  recall:', recall, '  F1:', f1)


    #早期終了(バリデーションデータでのスコアがいいものを採用)                   
    if best_score < f1:
        best_models = [copy.deepcopy(model),copy.deepcopy(netG),copy.deepcopy(netD)]
        best_score = f1

    return f1,best_models,best_score

def test(test_loader,model_l):
    # validation                                                                
    all_preds = list()
    all_labels = list()

    [model,netG,netD] = model_l
    model.eval()
    netG.eval()
    netD.eval()

    with torch.no_grad():
        for batch in test_loader:
            # 推論                                                              
            image, label = batch['x'].float().to('cuda'), batch['y'].float().to('cuda')
            pred = model(netG(image)+image)
            # ログ 
            pred = torch.where(pred[:,0]<pred[:,1],1,0)                                                             
            pred = pred.detach().cpu().numpy()
            #print(pred.shape)
            label = label.detach().cpu().numpy()

            [all_preds.append(i) for i in pred.ravel()]
            [all_labels.append(i) for i in label.ravel()]
    # 精度計算 
    cm = confusion_matrix(all_labels, all_preds)   
    #print(cm)
    tn, fp, fn, tp = cm.flatten()
    print('tp:', tp, '  fn:', fn, '  fp:', fp, '  tn:', tn)
    acc = (tp+tn)/(tn+fp+fn+tp)
    precision = tp/(tp+fp+1e-16)
    recall = tp/(tp+fn+1e-16)    
    f1 = 2*precision*recall/(precision+recall+1e-16)
    print('acc:', acc, '  precision:', precision, '  recall:', recall, '  F1:', f1)

    return acc,precision,recall,f1

def visualize(model_l,test_loader,save_dir,device):
    [model,netG,netD] = model_l
    model.eval()
    netG.eval()
    netD.eval()

    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            if i==0:
                noise_image = batch['x'].float().to(device)
                bs,w,h,c = noise_image.size()
                n = netG(noise_image)
                x = noise_image+n
                x_s = x.to('cpu').detach().numpy().copy()
                noise_images = noise_image.to('cpu').detach().numpy().copy()
                for j in range(bs):
                    cv2.imwrite(os.path.join(save_dir,os.path.join('capture',str(j)+'_denoise.png')), (255*x_s[j]).astype(np.uint8).transpose(1,2,0))
                    cv2.imwrite(os.path.join(save_dir,os.path.join('capture',str(j)+'_noise.png')), (255*noise_images[j]).astype(np.uint8).transpose(1,2,0))

            else:
                continue
            



def main(args):
    
    def fix_seed(seed):
        # random
        random.seed(seed)
        # Numpy
        np.random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    SEED = 3407
    fix_seed(SEED)

    #ハイパーパラメータ
    model_name = 'resnet101'
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir,'capture')):
        os.makedirs(os.path.join(save_dir,'capture'))
    #画像にノイズを加えるかの処理
    if args.noise:
        blur = False
        gaussian = False
        if args.noise_type=='blur' or args.noise_type=='both':
            blur = True
        if args.noise_type=='gaussian' or args.noise_type=='both':
            gaussian = True
    else:
        blur = False
        gaussian = False

    print(f'save_dir:{save_dir}, BLUR_flag:{blur}, GAUSS_flag:{gaussian}')

    #データの読み込み
    print('data loading..')
    dataset = dataloader.PAIR_SDNET(args.data_dir,model_name,blur=blur,gaussian=gaussian)
    train_size = int(len(dataset)*0.8)
    val_size = len(dataset)-train_size
    trainset,testset = torch.utils.data.random_split(dataset,[train_size, val_size])
    train_loader = DataLoader(trainset, batch_size = args.bs, shuffle = True, num_workers = 4,pin_memory=True)
    test_loader = DataLoader(testset, batch_size = args.bs, shuffle = False, num_workers = 4,pin_memory=True)

    #サンプルの個数を確認
    train_0 = 0
    train_1 = 0
    for batch in train_loader:
        label = batch['y']
        train_0 += torch.sum(label==0)
        train_1 += torch.sum(label==1)
    test_0 = 0
    test_1 = 0
    for batch in test_loader:
        label = batch['y']
        test_0 += torch.sum(label==0)
        test_1 += torch.sum(label==1)
    

    print(f'TRAIN 0:{train_0}, 1:{train_1} TEST 0:{test_0}, 1:{test_1}')

    print('model creating..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device',device)
    #モデル選択                                                                 
    if model_name == 'resnet101':
        model = models.RESNET101(2)
        model.load_state_dict(torch.load(args.model_path))
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        optimizer = torch.optim.SGD(model.module.model.parameters(),lr=0.001,momentum=0.99)
        #optimizer = torch.optim.Adam(model.model.parameters(),lr=0.0001)

        netG = models.Generator()
        #netG.load_state_dict(torch.load('./result/br-gan/netG.pth'))
        netD = models.Discriminator()
        #netD.load_state_dict(torch.load('./result/br-gan/netD.pth'))
        netG = torch.nn.DataParallel(netG).to(device)
        netD = torch.nn.DataParallel(netD).to(device)
        optim_G = torch.optim.SGD(netG.module.parameters(),lr=0.001,momentum=0.99)
        optim_D = torch.optim.SGD(netD.module.parameters(),lr=0.001,momentum=0.99)


    model_l = [model,netG,netD]
    optim_l = [optimizer,optim_G,optim_D]

    loss_func = nn.CrossEntropyLoss(weight=torch.tensor([train_1/train_size,train_0/train_size])).to(device)

    print('start training..')
    train_loss = []
    train_loss_G = []
    train_loss_D = []
    test_loss = []
    best_score = 0
    best_models = [copy.deepcopy(model),copy.deepcopy(netG),copy.deepcopy(netD)]
    for epoch in tqdm(range(1,1+args.max_epoch_GAN)):
        loss_G,loss_D = train_GAN(train_loader,model_l,optim_l,loss_func,train_size,epoch,device)
        train_loss_G.append(loss_G)
        train_loss_D.append(loss_D)
        f1,best_models,best_score = valid(test_loader,model_l,best_score,best_models)
        test_loss.append(f1)
        if epoch%10==0:
            visualize(model_l,test_loader,save_dir,device)
    
    #誤差関数のプロット
    fig = plt.figure()
    plt.plot(np.arange(len(train_loss_G)),np.array(train_loss_G),c='orange',label='train_G')
    plt.plot(np.arange(len(train_loss_D)),np.array(train_loss_D),c='blue',label='train_D')
    plt.plot(np.arange(len(test_loss)),np.array(test_loss),c='green',label='test(F1-score)')
    plt.title('Loss Value')
    plt.legend()
    fig.savefig(os.path.join(save_dir,"loss_GAN.png"))

    test_loss = []
    for epoch in tqdm(range(1,1+args.max_epoch_model)):
        loss = train_model(train_loader,model_l,optim_l,loss_func,train_size,epoch,device)
        train_loss.append(loss)
        f1,best_models,best_score = valid(test_loader,model_l,best_score,best_models)
        test_loss.append(f1)

    
    torch.save(best_models[0].module.to('cpu').state_dict(),os.path.join(save_dir,'model.pth'))
    torch.save(best_models[1].module.to('cpu').state_dict(),os.path.join(save_dir,'netG.pth'))
    torch.save(best_models[2].module.to('cpu').state_dict(),os.path.join(save_dir,'netD.pth'))
    best_models[0].to(device)
    best_models[1].to(device)
    best_models[2].to(device)

    acc,precision,recall,f1 = test(test_loader,best_models)
    #結果の記入
    res_file = open(os.path.join(save_dir,"result.txt"), mode="w")
    res_file.write("acc: "+str(acc)+'\n')
    res_file.write("precision: "+str(precision)+'\n')
    res_file.write("recall: "+str(recall)+'\n')
    res_file.write("F1-score: "+str(f1)+'\n')
        
    #誤差関数のプロット
    fig = plt.figure()
    plt.plot(np.arange(len(train_loss)),np.array(train_loss),c='orange',label='train(cross-entropy-loss)')
    plt.plot(np.arange(len(test_loss)),np.array(test_loss),c='green',label='test(F1-score)')
    plt.title('Loss Value')
    plt.legend()
    fig.savefig(os.path.join(save_dir,"loss.png"))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ベースラインの実験')
    parser.add_argument('data_dir', help='SDNETのデータディレクトリ')
    parser.add_argument('save_dir', help='結果の保存ディレクトリ')
    parser.add_argument('--model_path', help='classificationモデルの重みパス')
    parser.add_argument('--noise',action='store_true',help='inputの画像にノイズを入れるか')    
    parser.add_argument('--noise_type',type=str,default='',help='ノイズの種類/ blur,gaussian,both') 

    #学習パラメータ
    parser.add_argument('--max_epoch_model',type=int,default=50,help='classificationモデルのエポック数') 
    parser.add_argument('--max_epoch_GAN',type=int,default=100,help='GANのエポック数') 
    parser.add_argument('--bs',type=int,default=200,help='バッチサイズ数') 
    parser.add_argument('--lr',type=float,default=0.001,help='学習率') 

    args = parser.parse_args()   

    main(args)