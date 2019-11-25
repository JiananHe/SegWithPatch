import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.dice_loss import DiceLoss
from utils import setup_seed, organs_properties

num_organ = organs_properties['num_organ']
dropout_rate = 0.3
setup_seed(2018)


class VNet(nn.Module):
    def __init__(self, training, inchannel):
        """
        :param training: training or test
        :param inchannel: 网络最开始的输入通道数量
        """
        super(VNet, self).__init__()
        self.training = training

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(inchannel, 16, 3, 1, padding=1),
            nn.PReLU(16)
        )
        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )
        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )
        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )
        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )
        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )
        self.map = nn.Sequential(
            nn.Conv3d(32, num_organ + 1, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        encode1 = self.encoder_stage1(inputs) + inputs

        down1 = self.down_conv1(encode1)

        encode2 = self.encoder_stage2(down1) + down1
        encode2 = F.dropout(encode2, dropout_rate, self.training)

        down2 = self.down_conv2(encode2)

        encode3 = self.encoder_stage3(down2)
        encode3 = encode3 + down2
        encode3 = F.dropout(encode3, dropout_rate, self.training)

        down3 = self.down_conv3(encode3)

        encode4 = self.encoder_stage4(down3) + down3
        encode4 = F.dropout(encode4, dropout_rate, self.training)

        down4 = self.down_conv4(encode4)

        decode1 = self.decoder_stage1(encode4) + down4
        decode1 = F.dropout(decode1, dropout_rate, self.training)

        up1 = self.up_conv1(decode1)

        decode2 = self.decoder_stage2(torch.cat([up1, encode3], dim=1)) + up1
        decode2 = F.dropout(decode2, dropout_rate, self.training)

        up2 = self.up_conv2(decode2)

        decode3 = self.decoder_stage3(torch.cat([up2, encode2], dim=1)) + up2
        decode3 = F.dropout(decode3, dropout_rate, self.training)

        up3 = self.up_conv3(decode3)

        decode4 = self.decoder_stage4(torch.cat([up3, encode1], dim=1)) + up3

        outputs = self.map(decode4)

        return outputs


def get_net(is_training):
    # 网络参数初始化函数
    def init(module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(module.weight.data, 0.25)
            nn.init.constant_(module.bias.data, 0)

    net = VNet(is_training, 1)
    net.apply(init)

    return net


if __name__ == "__main__":
    net = get_net(True)
    net = net.cuda()
    # summary(net, (1, 48, 256, 256))

    ct = torch.randn((5, 1, 64, 64, 64)).cuda()
    seg = torch.randint(0, 13, (5, 64, 64, 64)).cuda()
    loss_func = DiceLoss()

    with torch.no_grad():
        out = net(ct)
        print(out.cpu().detach().numpy().shape)
        loss = loss_func(out, seg)
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