"""
label smooth
class weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from utils import *

num_organ = organs_properties['num_organ']


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    # tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        # tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        # tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn


class DiceLoss(nn.Module):
    def __init__(self, class_weight):
        super().__init__()

    def forward(self, predict, target, batch_weight):
        """
        计算多类别平均dice loss
        :param predict: tensor, 阶段一的输出 (B, 14, 64, 64, 64)
        :param target: numpy, 金标准 (B, 64, 64, 64)
        :return: loss
        """
        # 首先将金标准拆开
        (d, w, h) = target.shape[-3:]
        organs_target = np.zeros((target.shape[0], num_organ+1, d, w, h))
        for idx in range(num_organ+1):
            organs_target[:, idx, :, :, :] = (target == idx) + .0

        # organs_target = organs_target.cuda(predict.device.index).long()  # (B, Cls, *patch size)
        predict = F.softmax(predict, dim=1)
        organs_target = torch.from_numpy(organs_target).cuda(predict.device.index).long()

        loss_sum = 0.0
        organs_count = 0
        for idx in range(num_organ + 1):
            target_temp = organs_target[:, idx, :, :, :]
            if (target_temp == 0).all():  # 可能所有batch中的该器官都没有
                continue
            pred_temp = predict[:, idx, :, :, :]
            pred_temp = 0.9 - torch.relu(0.9 - pred_temp)
            org_dice = 2 * (torch.sum(pred_temp * target_temp, [1, 2, 3])) / \
                       (torch.sum(pred_temp, [1, 2, 3])
                        + torch.sum(target_temp, [1, 2, 3]) + 1e-6)
            org_loss = 1 - org_dice
            loss_sum += org_loss
            organs_count += 1

        loss_sum /= organs_count
        return loss_sum.mean()


if __name__ == "__main__":
    batch_size = 2
    class_num = 3
    loss_func = DiceLoss([1.0, 2.0, 3.0])
    ct = torch.randint(0, class_num, (batch_size, class_num, 5))
    seg = ct
    # print(ct.numpy())
    # print(seg.numpy())

    # pytorch
    print(loss_func(ct, seg))
