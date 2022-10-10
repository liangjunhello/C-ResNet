## Overview
Coco2014 is the only dataset for object detection task.\
Baseline: faster rcnn.\
Reference code:
https://github.com/pytorch/vision/tree/main/references/detection
https://github.com/pytorch/vision/tree/main/torchvision/models/detection

## Example

Examples for running train.py on terminal:
  - If you want to train C-ResNet15-A1 as backbone in a multi-gpu environment, run it by the command: `python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU --blockid 2 --layers 1,1,1,1`.
  - If you want to train C-ResNet18-A as backbone in a multi-gpu environment, run it by the command: `python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU - -blockid 3 -layers 1,2,1,1`.
  - If you want to train C-ResNet27-B2 as backbone in a multi-gpu environment, run it by the command: `python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU - -blockid 9 -layers 1,1,1,1`.
 
If you want to konw more details about running, you can read the source code:  `train.py`. 

## Dependencies

The script has been tested running under Python 3.7.4, with the following packages installed (along with their dependencies):

- `torch==1.11.0`
- `torchvision==0.12.0`

In addition, CUDA 10.2 have been used. We experimented on 4 Tesla A30s.

## Reference
If you make advantage of the C-resnet model in your research, please cite the following in your manuscript:

```
@article{
  
}
```

