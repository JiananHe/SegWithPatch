import torch
import torch.nn as nn
import numpy as np
from utils import organs_properties


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        """
        :param output: (B, 14, 64, 64, 64)
        :param target: (B, 64, 64, 64)
        """
        return self.loss(output, target)



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