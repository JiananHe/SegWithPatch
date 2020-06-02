import torch
import torch.nn as nn
import numpy as np
from utils import organs_properties
from utils import sum_tensor


class CELoss(nn.Module):
    def __init__(self, class_weight):
        super().__init__()
        self.class_weight = torch.tensor(class_weight).cuda().float()
        self.loss = nn.CrossEntropyLoss(weight=self.class_weight, reduction="none")

    def forward(self, output, target, batch_weight):
        """
        :param output: (B, 14, 64, 64, 64)原始网络输出，没有经过正则化
        :param target: (B, 64, 64, 64)
        """
        batch_weight = torch.tensor(batch_weight).cuda(output.device.index).float()
        assert isinstance(output, torch.Tensor)
        assert len(self.class_weight) == output.shape[1]
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).cuda(output.device.index).long()

        ce_loss = self.loss(output, target)  # (N, z, x, y)
        mean_axes = list(range(1, len(ce_loss.shape)))
        ce_loss = ce_loss.mean(dim=mean_axes)

        ce_loss = ce_loss * batch_weight
        return ce_loss.mean()


if __name__ == "__main__":
    batch_size = 2
    class_num = 3
    loss_func = CELoss([1.0, 2.0, 3.0])
    out = torch.randn((batch_size, class_num, 5, 5)).cuda()
    seg = torch.randint(0, class_num, (batch_size, 5, 5)).cuda()
    # print(ct.numpy())
    # print(seg.numpy())

    # pytorch
    print(loss_func(out, seg, [1] * 2))
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