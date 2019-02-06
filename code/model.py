import torch
import torchvision
from torch import nn
from torch.nn import functional as F


class ConvBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        # print(x.shape)
        return x


class FirstBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FirstBlock, self).__init__()
        self.conv1 = ConvBNRelu(in_ch, out_ch)
        self.conv2 = ConvBNRelu(out_ch, out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBNRelu(in_ch, out_ch)
        self.conv2 = ConvBNRelu(out_ch, out_ch)

    def forward(self, x):
        x = F.max_pool2d(x,kernel_size=2,stride=2)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, block_num=2):
        super(Encoder, self).__init__()
        layers = []
        layers += [ConvBNRelu(in_ch, out_ch)]
        for i in range(block_num-1):
            layers += [ConvBNRelu(out_ch, out_ch)]
        # layers += [nn.Dropout2d(0.5)]
        self.features = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x, indices = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True) 
        return x, indices

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, block_num=2):
        super(Decoder, self).__init__()
        layers = []
        layers += [ConvBNRelu(in_ch, out_ch)]
        for i in range(block_num-1):
            layers += [ConvBNRelu(out_ch, out_ch)]
        # layers += [nn.Dropout2d(0.5)]
        self.features = nn.Sequential(*layers)
        
    def forward(self, x, indices):
        x = F.max_unpool2d(x, indices=indices, kernel_size=2, stride=2)
        x = self.features(x)
        return x

class SegRoot(nn.Module):
    def __init__(self, width=8, depth=5, num_classes=2):
        super(SegRoot, self).__init__()
        chs = []
        for i in range(depth-1):
            chs.append(width * (2**i))
        chs.append(chs[-1])
        self.e_ch_info = [3,] + chs
        self.e_bl_info = [2,2,3,3]
        for _ in range(depth - 4):
            self.e_bl_info += [3,]
        self.d_ch_info = chs[::-1] + [4,]
        self.d_bl_info = self.e_bl_info[::-1]
        # using same setup with Unet
        if width == 4:
            self.e_ch_info = [3,4,8,16,32,64]
            self.d_ch_info = [64,32,16,8,4,4]
        self.num_classes = num_classes
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(1,len(self.e_ch_info)):
            self.encoders.append(Encoder(self.e_ch_info[i-1], self.e_ch_info[i], self.e_bl_info[i-1]))
            self.decoders.append(Decoder(self.d_ch_info[i-1], self.d_ch_info[i], self.d_bl_info[i-1]))
        
        # self.classifier = nn.Conv2d(self.d_ch_info[-1], num_classes, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(self.d_ch_info[-1], 1, 1)
        
    def forward(self, x):
        indices = []
        bs = x.shape[0]
        for i in range(len(self.e_bl_info)):
            x, ind = self.encoders[i](x)
            indices.append(ind)
            
        indices = indices[::-1]    
        for i in range(len(self.e_bl_info)):
            x = self.decoders[i](x, indices[i])
        
        x = self.classifier(x)
        # x = F.softmax(x,dim=1)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    x = torch.zeros((1, 3, 256, 256))
    net = SegRoot(8,5)
    print(net(x).shape)
