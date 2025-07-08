import torch.nn as nn
import torch
from torchvision.models import resnet18,resnet50

class EncoderBlock3D(nn.Module):
    '''
    Modified from Encoder implementation of LNE project (https://github.com/ouyangjiahong/longitudinal-neighbourhood-embedding)
    '''
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, pooling=nn.AvgPool3d):
        super(EncoderBlock3D, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)
        self.conv = nn.Sequential(
                        nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                        nn.BatchNorm3d(out_num_ch),
                        conv_act_layer,
                        nn.Dropout3d(dropout),
                        pooling(2))
        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)

class Encoder3D(nn.Module):
    def __init__(self, in_num_ch=1, num_block=4, inter_num_ch=16, kernel_size=3, conv_act='leaky_relu',  pooling=nn.AvgPool3d):
        super(Encoder3D, self).__init__()

        conv_blocks = []
        for i in range(num_block):
            if i == 0: # initial block
                conv_blocks.append(EncoderBlock3D(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0,  pooling=pooling))
            elif i == (num_block-1): # last block
                conv_blocks.append(EncoderBlock3D(inter_num_ch * (2 ** (i - 1)), inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0,  pooling=pooling))
            else:
                conv_blocks.append(EncoderBlock3D(inter_num_ch * (2 ** (i - 1)), inter_num_ch * (2 ** (i)), kernel_size=kernel_size, conv_act=conv_act, dropout=0,  pooling=pooling))

        self.conv_blocks = nn.Sequential(*conv_blocks)


    def forward(self, x):

        for cb in self.conv_blocks:
            x = cb(x)

        return x

class EncoderBlock2D(nn.Module):
    '''
    LSSL implementation from longitudinal-neighbourhood-embedding
    '''
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0,  pooling=nn.AvgPool2d):
        super(EncoderBlock2D, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        self.conv = nn.Sequential(
                        nn.Conv2d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                        nn.BatchNorm2d(out_num_ch),
                        conv_act_layer,
                        nn.Dropout2d(dropout),
                        pooling(2))

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)

class Encoder2D(nn.Module):
    def __init__(self, in_num_ch=1, num_block=4, inter_num_ch=16, kernel_size=3, conv_act='leaky_relu',  dropout=False, pooling=nn.AvgPool2d):
        super(Encoder2D, self).__init__()

        dropoutlist = [0, 0.1, 0.2, 0]

        conv_blocks = []
        for i in range(num_block):
            if i < 4 and dropout:
                dropout_ratio = dropoutlist[i]
            else:
                dropout_ratio = 0

            if i == 0: # initial block
                conv_blocks.append(EncoderBlock2D(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=dropout_ratio,  pooling=pooling))
            elif i == (num_block-1): # last block
                conv_blocks.append(EncoderBlock2D(inter_num_ch * (2 ** (i - 1)), inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=dropout_ratio,  pooling=pooling))
            else:
                conv_blocks.append(EncoderBlock2D(inter_num_ch * (2 ** (i - 1)), inter_num_ch * (2 ** (i)), kernel_size=kernel_size, conv_act=conv_act, dropout=dropout_ratio,  pooling=pooling))

        self.conv_blocks = nn.Sequential(*conv_blocks)


    def forward(self, x):

        for cb in self.conv_blocks:
            x = cb(x)

        return x

class CNNbasic2D(nn.Module):
    def __init__(self, inputsize=[64, 64], n_of_blocks=4, channels=3, initial_channel=16, pooling=nn.AvgPool2d, additional_feature=0):
        super(CNNbasic2D, self).__init__()

        self.feature_image = (torch.tensor(inputsize) / (2**(n_of_blocks)))
        self.feature_channel = initial_channel
        self.encoder = Encoder2D(in_num_ch=channels, num_block=n_of_blocks, inter_num_ch=initial_channel,  pooling=pooling)
        self.linear = nn.Linear((self.feature_channel * (self.feature_image.prod()).type(torch.int).item()) + additional_feature, 1, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], (self.feature_channel * (self.feature_image.prod()).type(torch.int).item()))
        y = self.linear(x)
        return y

class CNNbasic3D(nn.Module):
    def __init__(self, inputsize=[128, 128, 128], channels=1, n_of_blocks=4, initial_channel=16, pooling=nn.AvgPool3d, additional_feature=0):
        super(CNNbasic3D, self).__init__()

        self.feature_image = (torch.tensor(inputsize) / (2**(n_of_blocks)))
        self.feature_channel = initial_channel
        self.encoder = Encoder3D(in_num_ch=channels, num_block=n_of_blocks, inter_num_ch=initial_channel,  pooling=pooling)
        self.linear = nn.Linear((self.feature_channel * (self.feature_image.prod()).type(torch.int).item()) + additional_feature, 1, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], (self.feature_channel * (self.feature_image.prod()).type(torch.int).item()))
        y = self.linear(x)
        return y


class ResNet3DBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet3DBasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet183D(nn.Module):
    '''
    https://www.geeksforgeeks.org/deep-learning/resnet18-from-scratch-using-pytorch/
    '''
    def __init__(self, inputsize):
        super(ResNet183D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResNet3DBasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResNet3DBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNet3DBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNet3DBasicBlock, 512, 2, stride=2)
        
        self.feature_image = (torch.tensor(inputsize) / 16)
        self.feature_channel = 512
        self.linear = nn.Linear((self.feature_channel * (self.feature_image.prod()).type(torch.int).item()), 1, bias=False)
        # self.feature_image = (torch.tensor(inputsize) / (2**(n_of_blocks)))
        # self.feature_channel = initial_channel
        #self.linear = nn.Linear((self.feature_channel * (self.feature_image.prod()).type(torch.int).item()) + additional_feature, 1, bias=False)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear = nn.Linear(512, 1, bias=False)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(out.shape[0], (self.feature_channel * (self.feature_image.prod()).type(torch.int).item()))
        y = self.linear(out)
        return y
        
        # out = self.avgpool(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        # return out

def get_backbone(args=None):
    assert args != None, 'arguments are required for network configurations'
    # TODO args.optional_meta type should be list
    n_of_meta = len(args.optional_meta)

    backbone_name = args.backbone_name
    if backbone_name == 'cnn_3D':
        backbone = CNNbasic3D(inputsize=args.image_size, channels=args.image_channel, additional_feature = n_of_meta, initial_channel=args.inter_num_ch, n_of_blocks=args.num_block)
        linear = backbone.linear
        backbone.linear = nn.Identity()
    elif backbone_name == 'cnn_2D':
        backbone = CNNbasic2D(inputsize=args.image_size, channels=args.image_channel, additional_feature = n_of_meta, initial_channel=args.inter_num_ch, n_of_blocks=args.num_block)
        linear = backbone.linear
        backbone.linear = nn.Identity()
    elif backbone_name == 'resnet50_2D':
        backbone = resnet50()
        if args.image_channel != 3:
            backbone.conv1 = nn.Conv2d(args.image_channel, 64, 7, 2, 3, bias=False)
        linear = nn.Linear(2048 + n_of_meta, 1, bias=False)
        backbone.fc = nn.Identity()
    elif backbone_name == 'resnet18_2D':
        backbone = resnet18()
        if args.image_channel != 3:
            backbone.conv1 = nn.Conv2d(args.image_channel, 64, 7, 2, 3, bias=False)
        linear = nn.Linear(512 + n_of_meta, 1, bias=False)
        backbone.fc = nn.Identity()
    elif backbone_name == 'resnet18_3D':
        backbone = ResNet183D(inputsize=args.image_size)
        linear = backbone.linear
        backbone.linear = nn.Identity()
    else:
        raise NotImplementedError(f"{args.backbone_name} not implemented yet")
    
    return backbone, linear

class LILAC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone, self.linear = get_backbone(args)
        self.optional_meta = len(args.optional_meta)>0
    def forward(self, x1, x2, meta = None, return_f=False):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        f = f1 - f2
        if not self.optional_meta:
            if return_f:
                return self.linear(f), f1, f2
            else:
                return self.linear(f)
        else:
            m1, m2 = meta
            m = m1 - m2
            f = torch.concat((f, m), 1)
            if return_f:
                return self.linear(f), f1, f2
            else:
                return self.linear(f)

