# Cross-Resnet
Cross-Resnet ():

C-ResNet18-A            |  C-ResNet27-A2            |  C-ResNet27-B            |  C-ResNet27-C            |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](image/C-ResNet18-A.png)  |  ![](image/C-ResNet27-A2.png) |  ![](image/C-ResNet27-B.png) |  ![](image/C-ResNet27-C.png)

## Overview
Here we provide the code of nine cross resnet block in pytorch, along with the four datasets for training in paper . The repository is organised as follows:
- `data/` contains two datasets data (Caltech101 and Caltech256), and others' two datasets cifar10 and cifar100 can download from internet easily. Caltech101 source:http://www.vision.caltech.edu/Image_Datasets/Caltech101/ .
Caltech256 source:http://www.vision.caltech.edu/Image_Datasets/Caltech256/.
- `resnetimprove.py` contains the code of the  original resnet block (BasicBlock and Bottleneck) and nine cross resnet block ('C_BasicBlock_A1','C_BasicBlock_A','C_BasicBlock_A2','C_Bottleneck_C1','C_Bottleneck_C','c_Bottleneck_B','c_Bottleneck_B1','c_Bottleneck_B2','c_Bottleneck_B3');
- `main.py` running model training.

## Example
We use cifar10 for the default data set, and -d for changing the data set (0 Cifar10, 1 Cifar100, 2 Caltech101,3 Caltech256). c_BasicBlock_A1 is the default Cross Block, change the Cross Block with the -b command 
(0 BasicBlock,1 Bottleneck, 2 C_BasicBlock_A1 ,3 C_BasicBlock_A, 4 C_BasicBlock_A2 , 5 C_Bottleneck_C1 , 6 C_Bottleneck_C , 7 c_Bottleneck_B , 8 c_Bottleneck_B1 , 9 c_Bottleneck_B2 , 10 c_Bottleneck_B3 ). 
change the stack structure of Cross blocks with -l command, e.g. -l 2,2,2,2 .

Examples for running main.py on terminal:
  - If you want to train C-ResNet15-A1 on Cifar10, run it by the command: `python main.py -l 1,1,1,1`.
  - If you want to train C-ResNet18-A on Cifar10, run it by the command: `python main.py -b 3 -l 1,2,1,1`.
  - If you want to train C-ResNet27-A2 on Cifar100, run it by the command: `python main.py -d 1 -b 4 -l 2,2,2,2`.
  - If you want to train C-ResNet27-B on Caltech101, run it by the command: `python main.py -d 2 -b 7 -l 1,1,1,1`.
  - If you want to train C-ResNet27-C on Caltech256, run it by the command: `python main.py -d 3 -b 6 -l 2,2,2,2`.
  
If you want to konw more details about running, you can read the source code:  `main.py`. 

## Dependencies

The script has been tested running under Python 3.7.4, with the following packages installed (along with their dependencies):

- `numpy==1.18.1`
- `torch==1.5.0`
- `torchvision==0.6.0`

In addition, CUDA 10.2 and cuDNN 7.6.5 have been used. We experimented on Tesla V100.

## Reference
If you make advantage of the C-resnet model in your research, please cite the following in your manuscript:

```
@article{
  
}
```


