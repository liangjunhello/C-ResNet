import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from datetime import datetime
import resnetimprove as resnet
import os
import numpy as np 
import time
import torch.backends.cudnn as cudnn
import warnings
from torch.utils.data import Dataset
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
parser = argparse.ArgumentParser(description='resnet Training')
parser.add_argument('-d', '--datasetid',  default=0, type=int,
                    help='the dataset to train (0 for cifar10, 1 cifar100, 2 caltech101, 3 caltech256')
parser.add_argument('-b', '--blockid', default=2,type=int,               
                    help='0 BasicBlock, 1 Bottleneck, 2 C_BasicBlock_A1, '+
                    '3 C_BasicBlock_A, 4 C_BasicBlock_A2, 5 C_Bottleneck_C1, 6 C_Bottleneck_C'+
                    '7 c_Bottleneck_B, 8 c_Bottleneck_B1, 9 c_Bottleneck_B2, 10 c_Bottleneck_B3')

parser.add_argument('-l', '--layers',default='2,2,2,2',type=str,
                    help='num of layer,like resnet18:2,2,2,2 resnet34:3,4,6,3')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('-lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--adjust-lr-epoch', default=150, type=int,
                    help='learning rate adjust by epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--save-epoch', default=20, type=int,
                    help='epoch to save model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.such as 0,1,2 means make 0,1,2 three gpus available.')
parser.add_argument('--save-path',default='./model_save',type=str)
parser.add_argument('--pretrain-model',default=None,type=str,
                    help='pretrain_model file name(the file save under ./model_save)')


data_path="./data/"
args = parser.parse_args()

    
class CalDataset(Dataset):
    def __init__ (self,text_path,path,transform=None,target_transform=None):
        fh=open(text_path,'r')
        imgs=[]
        for line in fh:
            line = line.rstrip()  
            words=line.split()    
            imgs.append((words[0],int(words[1])))
            self.imgs=imgs
            self.transform = transform
            self.target_transform = target_transform
            self.path=path
        
    def __getitem__ (self,index):      
        fn,label = self.imgs[index]   
        fn = self.path + fn
        img = Image.open(fn).convert("RGB")  
        if self.transform is not None :
            img = self.transform(img)
        return img,label
    def __len__ (self):
        return len(self.imgs)



def net_construct(blockid,layers,num_class):

    if(blockid==0):return resnet.ResNet(resnet.BasicBlock, layers,num_classes=num_class)
    elif(blockid==1):return resnet.ResNet(resnet.Bottleneck, layers,num_classes=num_class)
    elif(blockid==2):return resnet.ResNet(resnet.c_BasicBlock_A1, layers,num_classes=num_class)
    elif(blockid==3):return resnet.ResNet(resnet.c_BasicBlock_A, layers,num_classes=num_class)
    elif(blockid==4):return resnet.ResNet(resnet.c_BasicBlock_A2, layers,num_classes=num_class)
    elif(blockid==5):return resnet.ResNet(resnet.c_Bottleneck_C1, layers,num_classes=num_class)
    elif(blockid==6):return resnet.ResNet(resnet.c_Bottleneck_C, layers,num_classes=num_class)
    elif(blockid==7):return resnet.ResNet(resnet.c_Bottleneck_B, layers,num_classes=num_class)
    elif(blockid==8):return resnet.ResNet(resnet.c_Bottleneck_B1, layers,num_classes=num_class)
    elif(blockid==9):return resnet.ResNet(resnet.c_Bottleneck_B2, layers,num_classes=num_class)
    elif(blockid==10):return resnet.ResNet(resnet.c_Bottleneck_B3, layers,num_classes=num_class)
    else:
        args.blockid=0
        return resnet.ResNet(resnet.BasicBlock, layers,num_classes=num_class)
    
def dataset_choose(dataset_id):
    train_set=None
    test_set=None
    if(dataset_id==0):
        train_set = CIFAR10(data_path, train=True, transform=transform_train,download=True)
        test_set = CIFAR10(data_path, train=False, transform=transform_test,download=True)        
    elif(dataset_id==1):
        train_set = CIFAR100(data_path, train=True, transform=transform_train,download=True)
        test_set = CIFAR100(data_path, train=False, transform=transform_test,download=True)
    elif(dataset_id==2):
        base_path=data_path+"Caltech101/"
        train_path=base_path+"dataset-train.txt"
        val_path=base_path+"dataset-val.txt"
        train_set = CalDataset(text_path=train_path,path=base_path,transform=transform_train,target_transform=None) 
        test_set= CalDataset(text_path=val_path,path=base_path,transform=transform_test,target_transform=None)
    elif(dataset_id==3):
        base_path=data_path+"Caltech256/"
        train_path=base_path+"dataset-train.txt"
        val_path=base_path+"dataset-val.txt"
        train_set = CalDataset(text_path=train_path,path=base_path,transform=transform_train,target_transform=None) 
        test_set= CalDataset(text_path=val_path,path=base_path,transform=transform_test,target_transform=None)
    else:
        args.datasetid=0
        train_set = CIFAR10(data_path, train=True, transform=transform_train,download=True)
        test_set = CIFAR10(data_path, train=False, transform=transform_test,download=True)
    return train_set,test_set
    
#Data preparation for cifar10. Borrowed from
#https://github.com/kuangliu/pytorch-cifar/blob/master/main.py    
transform_train = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
dataset_numclass=[10,100,101,256]

dataset_name=['cifar10','cifar100','caltech101','caltech256']
block_name=['BasicBlock','Bottleneck','C_BasicBlock_A1','C_BasicBlock_A', 'C_BasicBlock_A2', 'C_Bottleneck_C1', 'C_Bottleneck_C',
            'c_Bottleneck_B', 'c_Bottleneck_B1','c_Bottleneck_B2','c_Bottleneck_B3'] 


    
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def adjust_learning_rate(optimizer, epoch, lr,lr_epoch):
    """Sets the learning rate to the initial LR decayed"""
    lr *= (0.1 ** (epoch // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def train(net, train_data, valid_data, num_epochs, optimizer, criterion,LR,lr_epoch):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        print(epoch,"time train ")
        adjust_learning_rate(optimizer, epoch, LR,lr_epoch)
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        
        if epoch%args.save_epoch==0: 
            torch.save(net.state_dict(),
                       "./model_save/{}_{}_{}.pt".format(block_name[args.blockid],args.layers,dataset_name[args.datasetid]))  
        
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    with torch.no_grad():
                        im = Variable(im.cuda())
                        label = Variable(label.cuda())
                        output = net(im)
                        loss = criterion(output, label)
                        valid_loss += loss.item()
                        valid_acc += get_acc(output, label)
                else:
                    with torch.no_grad():
                        im = Variable(im)
                        label = Variable(label)
                        output = net(im)
                        loss = criterion(output, label)
                        valid_loss += loss.item()
                        valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))

        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)



def main():    
    if args.gpu is not None:        
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. ')
        
    train_set,test_set=dataset_choose(args.datasetid)
    train_data=torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,num_workers=args.workers)
    test_data=torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)
    net=net_construct(args.blockid,[int(s) for s in args.layers.split(',')],dataset_numclass[args.datasetid])
   
    
    net=nn.DataParallel(net)  
    if args.pretrain_model is not None:
        pretrained_net = torch.load('./model_save/'+args.pretrain_model)
        net.load_state_dict(pretrained_net)
    criterion = nn.CrossEntropyLoss()  
    LR=args.lr
    epoch=args.epochs
    lr_epoch=args.adjust_lr_epoch
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=args.momentum, weight_decay=args.weight_decay) 
    print("Start training for block class:%s dataset:%s layers:%s "%(block_name[args.blockid],dataset_name[args.datasetid],args.layers))
    train(net, train_data, test_data, epoch, optimizer, criterion,LR,lr_epoch)

if __name__ == '__main__':
    main()
