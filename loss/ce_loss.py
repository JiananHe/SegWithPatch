import torch
import torch.nn as nn
import numpy as np
from utils import organs_properties
from utils import sum_tensor


class CELoss(nn.Module):
    def __init__(self, class_weight):
        super().__init__()
        self.weight = torch.tensor(class_weight).cuda().float()
        self.loss = nn.CrossEntropyLoss(weight=self.weight, reduction="none")

    def forward(self, output, target, batch_weight):
        """
        :param output: (B, 14, 64, 64, 64)原始网络输出，没有经过正则化
        :param target: (B, 64, 64, 64)
        """
        batch_weight = torch.tensor(batch_weight).cuda(output.device.index).float()
        assert len(self.weight) == output.shape[1]
        ce_loss = self.loss(output, target)  # (N, z, x, y)
        ce_loss = ce_loss.mean(dim=[1, 2, 3])

        ce_loss = ce_loss * batch_weight
        return ce_loss.mean()


if __name__ == "__main__":
    loss_func = CELoss()
    ct = torch.randn((1, 3, 2))
    seg = torch.randint(0, 3, (1, 2))
    print(ct.numpy())
    print(seg.numpy())

    # # pytorch
    # print(loss_func(ct, seg))
    # # numpy
    # loss = .0
    # ct = ct.numpy()
    # seg = seg.numpy()
    # for i in range(ct.shape[0]):  # for every sample
    #     c = ct[i]
    #     s = seg[i]
    #     np.reshape()
    #     for cls in range(seg.shape[0]):  # for every class
    #         loss += np.log(c[])
    #         np.swapaxes()