"""
label smooth
class weight
"""

import torch
import torch.nn as nn
from utils import organs_properties

num_organ = organs_properties['num_organ']
organs_weight = organs_properties['organs_weight']


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        """
        计算多类别平均dice loss
        :param predict: 阶段一的输出 (B, 14, 64, 64, 64)
        :param target: 金标准 (B, 64, 64, 64)
        :return: loss
        """
        # 首先将金标准拆开
        (d, w, h) = target.size()[-3:]
        organs_target = torch.zeros(target.size(0), num_organ, d, w, h)
        for idx in range(num_organ):
            organs_target[:, idx, :, :, :] = (target == idx+1) + .0

        organs_target = organs_target.cuda() # (B, 13, 64, 64, 64)

        loss_sum = 0.0
        organs_count = 0
        for idx in range(1, num_organ+1):
            target_temp = organs_target[:, idx - 1, :, :, :]
            if (target_temp == 0).all():  # 可能所有batch中的该器官都没有
                continue
            pred_temp = predict[:, idx, :, :, :]
            pred_temp = 0.9-torch.relu(0.9-pred_temp)
            org_dice = 2 * (torch.sum(pred_temp * target_temp, [1, 2, 3])) / \
                           (torch.sum(pred_temp, [1, 2, 3])
                            + torch.sum(target_temp, [1, 2, 3]) + 1e-6)
            org_loss = organs_weight[idx-1] * (1 - org_dice)
            loss_sum += org_loss
            organs_count += 1

        loss_sum /= organs_count

        return loss_sum.mean()
