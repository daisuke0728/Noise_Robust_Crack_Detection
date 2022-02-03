import os,sys,time
import glob
import random
import argparse
from tqdm import tqdm
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

def train(train_loader,model,optimizer,loss_func,train_size,device):
    # training                                                                  
    all_loss = list()

    model.train()
    for batch in train_loader:
        image, label = batch['x'].float().to(device), batch['y'].to(device)
        bs,w,h,c = image.size()
        pred = model(image)
        # 損失計算                                                              
        loss = loss_func(pred, label)
        # 逆伝播                                                                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ログ                                                                  
        all_loss.append(bs*loss.item())

    train_loss = sum(all_loss)/train_size
    print()
    print('Train Loss:',train_loss)
    
    return train_loss

def valid(valid_loader,model,best_score,best_model):
    # validation                                                                
    all_preds = list()
    all_labels = list()
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            # 推論                                                              
            image, label = batch['x'].float().to('cuda'), batch['y'].float().to('cuda')
            pred = model(image)
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
        best_model = copy.deepcopy(model)
        best_score = f1

    return f1,best_model,best_score

def test(test_loader,model):
    # validation                                                                
    all_preds = list()
    all_labels = list()
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # 推論                                                              
            image, label = batch['x'].float().to('cuda'), batch['y'].float().to('cuda')
            pred = model(image)
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

    #シード値の固定
    SEED = 3407
    fix_seed(SEED)

    #ハイパーパラメータ
    model_name = 'resnet101'
    max_epoch = args.max_epoch
    save_dir = args.save_dir
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
    dataset = dataloader.SDNET(args.data_dir,model_name,blur=blur,gaussian=gaussian)
    train_size = int(len(dataset)*0.8)
    val_size = len(dataset)-train_size
    trainset,testset = torch.utils.data.random_split(dataset,[train_size, val_size])
    train_loader = DataLoader(trainset, batch_size = args.bs, shuffle = True, num_workers = 4,pin_memory=True)
    test_loader = DataLoader(testset, batch_size = args.bs, shuffle = False, num_workers = 4,pin_memory=True)

    #サンプルの個数を確認
    train_0,train_1,test_0,test_1 = 0,0,0,0
    for batch in train_loader:
        label = batch['y']
        train_0 += torch.sum(label==0)
        train_1 += torch.sum(label==1)
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
        model = models.RESNET101(2).to(device)
        optimizer = torch.optim.SGD(model.model.parameters(),lr=args.lr,momentum=0.99)
        #optimizer = torch.optim.Adam(model.model.parameters(),lr=0.0001)

    loss_func = nn.CrossEntropyLoss(weight=torch.tensor([train_1/train_size,train_0/train_size])).to(device)

    print('start training..')
    train_loss = []
    test_loss = []
    best_score = 0
    best_model = copy.deepcopy(model)
    for epoch in tqdm(range(max_epoch)):
        loss = train(train_loader,model,optimizer,loss_func,train_size,device)
        train_loss.append(loss)
        f1,best_model,best_score = valid(test_loader,model,best_score,best_model)
        test_loss.append(f1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(best_model.to('cpu').state_dict(),os.path.join(save_dir,'model.pth'))
    best_model.to(device)
    acc,precision,recall,f1 = test(test_loader,best_model)
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
    parser.add_argument('--noise',action='store_true',help='inputの画像にノイズを入れるか')    
    parser.add_argument('--noise_type',type=str,default='',help='ノイズの種類/ blur,gaussian,both') 

    #学習パラメータ
    parser.add_argument('--max_epoch',type=int,default=50,help='エポック数') 
    parser.add_argument('--bs',type=int,default=50,help='バッチサイズ数') 
    parser.add_argument('--lr',type=float,default=0.001,help='学習率') 

    args = parser.parse_args()   

    main(args)