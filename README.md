# Cross-Resnet
Cross-Resnet (Veličković *et al.*, ICLR 2018): [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)

cross_BasicBlock_1            |  Cross_BasicBlock_2
:-------------------------:|:-------------------------:
![](https://github.com/FrankXu0808/Cross_resnet/blob/master/readme_images/1-198x1200.png?raw=true)  |  ![](https://github.com/FrankXu0808/Cross_resnet/blob/master/readme_images/1-198x1200.png?raw=true)

## Overview
Here we provide the code of four cross resnet block in pytorch, along with the four datasets for training in paper . The repository is organised as follows:
- `data/` contains two datasets data (Caltech101 and Caltech256), and others' two datasets cifar10 and cifar100 can download from internet easily;
- `resnetimprove.py` contains the code of the  original resnet block (BasicBlock and Bottleneck) and four cross resnet block (Cross_BasicBlock_1, Cross_BasicBlock_2, Cross_Bottleneck_1, Cross_Bottleneck_2);
- `train.py` contains Code related to training;
- `main.py` running model training.



## Dependencies

The script has been tested running under Python 3.7.4, with the following packages installed (along with their dependencies):

- `numpy==1.18.1`
- `torch==1.5.0`
- `torchvision==0.6.0`

In addition, CUDA 10.2 and cuDNN 7.6.5 have been used. We experimented on four Tesla V100.

## Reference
If you make advantage of the GAT model in your research, please cite the following in your manuscript:

```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={accepted as poster},
}
```

## License
SCNU
