import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.dice_loss import DiceLoss
from loss.ce_loss import CELoss
from loss.dice_ce_loss import DC_and_CE_loss
import torchvision
import numpy as np
import sys
sys.path.append("../")
from utils import setup_seed, organs_properties, network_configure


num_organ = organs_properties['num_organ']
kernel_sizes = network_configure['kernel_sizes']
features_channels = network_configure['features_channels']
down_strides = network_configure['down_strides']
setup_seed(2020)


class StageLayer(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size):
        super(StageLayer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size, 1, padding=tuple(i//2 for i in kernel_size)),
            nn.InstanceNorm3d(outchannel),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(outchannel, outchannel, kernel_size, 1, padding=tuple(i//2 for i in kernel_size)),
            nn.InstanceNorm3d(outchannel),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TDUNet(nn.Module):
    def __init__(self, inchannel=1):
        """
        :param inchannel: 网络最开始的输入通道数量
        """
        super(TDUNet, self).__init__()

        self.encode1 = StageLayer(inchannel, features_channels[0], kernel_sizes[0])
        self.down1 = nn.Conv3d(features_channels[0], features_channels[1],
                               kernel_size=down_strides[0], stride=down_strides[0])
        self.encode2 = StageLayer(features_channels[1], features_channels[1], kernel_sizes[1])
        self.down2 = nn.Conv3d(features_channels[1], features_channels[2],
                               kernel_size=down_strides[1], stride=down_strides[1])
        self.encode3 = StageLayer(features_channels[2], features_channels[2], kernel_sizes[2])
        self.down3 = nn.Conv3d(features_channels[2], features_channels[3],
                               kernel_size=down_strides[2], stride=down_strides[2])
        self.encode4 = StageLayer(features_channels[3], features_channels[3], kernel_sizes[3])
        self.down4 = nn.Conv3d(features_channels[3], features_channels[4],
                               kernel_size=down_strides[3], stride=down_strides[3])
        self.encode5 = StageLayer(features_channels[4], features_channels[4], kernel_sizes[4])

        self.up1 = nn.ConvTranspose3d(features_channels[-1], features_channels[-2],
                                      kernel_size=down_strides[-1], stride=down_strides[-1])
        self.decode1 = StageLayer(2 * features_channels[-2], features_channels[-2], kernel_sizes[-2])
        self.up2 = nn.ConvTranspose3d(features_channels[-2], features_channels[-3],
                                      kernel_size=down_strides[-2], stride=down_strides[-2])
        self.decode2 = StageLayer(2 * features_channels[-3], features_channels[-3], kernel_sizes[-3])
        self.up3 = nn.ConvTranspose3d(features_channels[-3], features_channels[-4],
                                      kernel_size=down_strides[-3], stride=down_strides[-3])
        self.decode3 = StageLayer(2 * features_channels[-4], features_channels[-4], kernel_sizes[-4])
        self.out1 = nn.Conv3d(features_channels[-4], num_organ+1, kernel_size=1)

        self.up4 = nn.ConvTranspose3d(features_channels[-4], features_channels[-5],
                                      kernel_size=down_strides[-4], stride=down_strides[-4])
        self.decode4 = StageLayer(2 * features_channels[-5], features_channels[-5], kernel_sizes[-5])
        self.out2 = nn.Conv3d(features_channels[-5], num_organ+1, kernel_size=1)

    def forward(self, x):
        encode1 = self.encode1(x)
        down = self.down1(encode1)
        encode2 = self.encode2(down)
        down = self.down2(encode2)
        encode3 = self.encode3(down)
        down = self.down3(encode3)
        encode4 = self.encode4(down)
        down = self.down4(encode4)
        x = self.encode5(down)

        outputs = []
        x = self.up1(x)
        x = self.decode1(torch.cat([x, encode4], dim=1))
        x = self.up2(x)
        x = self.decode2(torch.cat([x, encode3], dim=1))
        x = self.up3(x)
        x = self.decode3(torch.cat([x, encode2], dim=1))
        out1 = self.out1(x)
        outputs.append(out1)

        x = self.up4(x)
        x = self.decode4(torch.cat([x, encode1], dim=1))
        out2 = self.out2(x)
        outputs.append(out2)

        return outputs


def get_net(in_channel):
    # 网络参数初始化函数
    def init(module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(module.weight, a=0.01)
            nn.init.constant_(module.bias, 0)

    net = TDUNet(in_channel)
    net.apply(init)

    return net


if __name__ == "__main__":
    # torchvision.models.vgg11_bn()
    net = get_net(1)
    net.cuda()
    # summary(net, (1, 48, 256, 256))

    ct = torch.randn((2, 1, 48, 128, 128)).cuda()
    seg = np.random.randint(0, 13, (2, 48, 128, 128))
    loss_func = DC_and_CE_loss([1.0]*(num_organ+1))

    with torch.no_grad():
        out = net(ct)
        print(out[0].cpu().detach().numpy()[0, :, 4,5,8])
        print(np.sum(out[0].cpu().detach().numpy()[0, :, 4,5,8]))
        print(np.argmax(out[0].cpu().detach().numpy()[0, :, 4,5,8]))
        print(out[0].cpu().detach().numpy().shape)
        print(out[1].cpu().detach().numpy().shape)

        loss = loss_func(out, seg, [1.0, 0.5])
        print(loss.item())



    # 计算参数个数
    count = .0
    for item in net.modules():
        if isinstance(item, nn.Conv3d) or isinstance(item, nn.ConvTranspose3d):
            count += (item.weight.size(0) * item.weight.size(1) *
                      item.weight.size(2) * item.weight.size(3) * item.weight.size(4))

            if item.bias is not None:
                count += item.bias.size(0)

        elif isinstance(item, nn.PReLU):
            count += item.num_parameters

    print("number of parameters: ", count)
