# Cross-Resnet
Cross-Resnet ():

C-ResNet18-A            |  C-ResNet27-A2            |  C-ResNet27-B            |  C-ResNet27-C            |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](image/C-ResNet18-A.png)  |  ![](image/C-ResNet27-A2.png) |  ![](image/C-ResNet27-B.png) |  ![](image/C-ResNet27-C.png)

## Overview
Here we provide the code of four cross resnet block in pytorch, along with the four datasets for training in paper . The repository is organised as follows:
- `data/` contains two datasets data (Caltech101 and Caltech256), and others' two datasets cifar10 and cifar100 can download from internet easily. Caltech101 source:http://www.vision.caltech.edu/Image_Datasets/Caltech101/ .
Caltech256 source:http://www.vision.caltech.edu/Image_Datasets/Caltech256/.
- `resnetimprove.py` contains the code of the  original resnet block (BasicBlock and Bottleneck) and nine cross resnet block ('C_BasicBlock_A1','C_BasicBlock_A','C_BasicBlock_A2','C_Bottleneck_C1','C_Bottleneck_C','c_Bottleneck_B','c_Bottleneck_B1','c_Bottleneck_B2','c_Bottleneck_B3');
- `main.py` running model training.



## Dependencies

The script has been tested running under Python 3.7.4, with the following packages installed (along with their dependencies):

- `numpy==1.18.1`
- `torch==1.5.0`
- `torchvision==0.6.0`

In addition, CUDA 10.2 and cuDNN 7.6.5 have been used. We experimented on four Tesla V100.

## Reference
If you make advantage of the C-resnet model in your research, please cite the following in your manuscript:

```
@article{
  
}
```


