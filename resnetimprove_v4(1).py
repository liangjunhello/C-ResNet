'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch.nn as nn
import torch.nn.functional as F


#original block
class BasicBlock(nn.Module): 
    
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module): 
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


#################################################################
        ###########################################
#Define C-BasicBlock
 #C-resnet15-A1-1111
class c_BasicBlock_A1(nn.Module):  
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(c_BasicBlock_A1, self).__init__()
        self.flag=stride
        if in_planes != self.expansion*planes:
            self.flag=2
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion*planes: 
            self.conv4 = nn.Conv2d(in_planes, planes, 1, stride=stride)
            self.conv1tov3 = nn.Conv2d(planes, planes, 1, stride=1)

    def forward(self, x):  
        out1 = F.relu(self.bn1(self.conv1(x)), True)
        out2=F.relu(self.bn2(self.conv2(out1)))
        if self.flag != 1:  
            x=self.conv4(x)
            out1=self.conv1tov3(out1)
        out2 = out2+x
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out3 = out3+out1

        return out3



#  C-resnet18-A 1211
class c_BasicBlock_A(nn.Module): #  
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(c_BasicBlock_A, self).__init__()
        self.flag=stride
        if in_planes != self.expansion*planes:
            self.flag=2
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion*planes: 
            self.conv4 = nn.Conv2d(in_planes, planes, 1, stride=stride)

    def forward(self, x):  

        out1= F.relu(self.bn1(self.conv1(x)), True)
        out2=F.relu(self.bn2(self.conv2(out1)))
        if self.flag != 1:  
            x=self.conv4(x)
        out2 = out2+x
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out3 = out3+out1

        return out3
 
  #C-resnet27-A2-2222
class c_BasicBlock_A2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(c_BasicBlock_A2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        
        self.shortcut_1 = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride))

    def forward(self, x):  
        out1 = F.relu(self.bn1(self.conv1(x)), True)
        out2=F.relu(self.bn2(self.conv2(out1)))
        out2 = out2+self.shortcut_1(x)
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out3 = out3+out1

        return out3
    

####################################################################
        ############################################
   #C-resnet27-C1-2222       
class c_Bottleneck_C1(nn.Module):  
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(c_Bottleneck_C1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut_1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,  bias=False)
        self.shortcut_2 =nn.Conv2d(planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):

        out1=F.relu(self.bn1(self.conv1(x)))
        out2 =F.relu(self.bn2(self.conv2(out1)))
        x= self.shortcut_1(x)
        out1=self.shortcut_2(out1)
        out2=out2+x
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out3 = out1+out3
        return out3

#
  #C-resnet27-C-2222
class c_Bottleneck_C(nn.Module):   
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(c_Bottleneck_C, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*planes)
        self.conv3 = nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut_1=nn.Sequential()
        self.shortcut_2 =nn.Conv2d(planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        if stride != 1 or in_planes != self.expansion*planes:           
            self.shortcut_1 =nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,  bias=False))
    
    def forward(self, x):

        out1=F.relu(self.bn1(self.conv1(x)))
        out2 =F.relu(self.bn2(self.conv2(out1)))
        x= self.shortcut_1(x)
        out1=self.shortcut_2(out1)     
        out2=out2+x
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out3 = out1+out3

        return out3


  #C-resnet27-B-1111  or C-resnet39-B-1221   one dashed line in a Bottleneck block
class c_Bottleneck_B(nn.Module):  
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(c_Bottleneck_B, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*planes)
        self.conv3 = nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        self.conv4 = nn.Conv2d(self.expansion*planes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.expansion*planes)
        self.conv6 = nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.expansion*planes)
                 
        self.shortcut_1 =nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,  bias=False)

    
    def forward(self, x):
        out1=F.relu(self.bn1(self.conv1(x)))
        out2 =F.relu(self.bn2(self.conv2(out1)))
        out3 =F.relu(self.bn3(self.conv3(out2)))
        out4 =F.relu(self.bn4(self.conv4(out3)))   
        out4=out4+ self.shortcut_1(x)
        out5 =F.relu(self.bn5(self.conv5(out4)))
        out5=out5+out2       
        out6 =F.relu(self.bn6(self.conv6(out5)))     
        out6 =out6+out3  
    
        return out6     
    
    
    
  #C-resnet27-B1-1111   one dashed line (the first line) in a Bottleneck block except for three solid lines in the first bottleneck block.
#Minimum number of dashed lines
class c_Bottleneck_B1(nn.Module):   
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(c_Bottleneck_B1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*planes)
        self.conv3 = nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        self.conv4 = nn.Conv2d(self.expansion*planes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.expansion*planes)
        self.conv6 = nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.expansion*planes)
        
        self.shortcut_1=nn.Sequential()
        if stride != 1:                   
            self.shortcut_1 =nn.Conv2d(planes, planes, kernel_size=1, stride=stride,  bias=False)
    
    def forward(self, x):
        out1=F.relu(self.bn1(self.conv1(x)))
        out2 =F.relu(self.bn2(self.conv2(out1)))
        out3 =F.relu(self.bn3(self.conv3(out2)))
        out4 =F.relu(self.bn4(self.conv4(out3)))
        
        out4=out4+  self.shortcut_1(out1)
        out5 =F.relu(self.bn5(self.conv5(out4)))
        out5=out5+out2  
        out6 =F.relu(self.bn6(self.conv6(out5)))
        out6 =out6+out3
        return out6  
    
    
  #C-resnet27-B2-1111,  two dashed lines (the first two lines) in a Bottleneck block     
class c_Bottleneck_B2(nn.Module):  
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(c_Bottleneck_B2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*planes)
        self.conv3 = nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        self.conv4 = nn.Conv2d(self.expansion*planes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.expansion*planes)
        self.conv6 = nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.expansion*planes)
        
         #two dashed lines
        self.shortcut_1 =nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,  bias=False)
        self.shortcut_2 =nn.Sequential(nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, stride=1,  bias=False))
        
    
    def forward(self, x):
        out1=F.relu(self.bn1(self.conv1(x)))
        out2 =F.relu(self.bn2(self.conv2(out1)))
        out3 =F.relu(self.bn3(self.conv3(out2)))
        out4 =F.relu(self.bn4(self.conv4(out3)))
        
        out4=out4+ self.shortcut_1(x) 
        out5 =F.relu(self.bn5(self.conv5(out4)))
        out5=out5+self.shortcut_2(out2)  
        out6 =F.relu(self.bn6(self.conv6(out5)))
        out6 =out6+out3  
        return out6 
    
    
    
  #C-resnet27-B3-1111,  three dashed lines (all lines are dashed) in a Bottleneck block        
class c_Bottleneck_B3(nn.Module):   
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(c_Bottleneck_B3, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*planes)
        self.conv3 = nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        self.conv4 = nn.Conv2d(self.expansion*planes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.expansion*planes)
        self.conv6 = nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.expansion*planes)
           
        #three dashed lines
        self.shortcut_1 =nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,  bias=False)
        self.shortcut_2 =nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, stride=1,  bias=False)
        self.shortcut_3 =nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, stride=1,  bias=False)
    
    def forward(self, x):

        out1=F.relu(self.bn1(self.conv1(x)))
        out2 =F.relu(self.bn2(self.conv2(out1)))
        out3 =F.relu(self.bn3(self.conv3(out2)))
        out4 =F.relu(self.bn4(self.conv4(out3)))
        
        out4=out4+ self.shortcut_1(x)
        out5 =F.relu(self.bn5(self.conv5(out4)))
        out5=out5+self.shortcut_2(out2)  
        out6 =F.relu(self.bn6(self.conv6(out5))) 
        out6 =out6+self.shortcut_2(out3) 
        return out6      

#####################################################################################

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv0= nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out=self.conv0(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out





