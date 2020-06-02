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
organs_weight = organs_properties['organs_weight']


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
    def __init__(self, class_weight, do_bg=True, smooth=1e-5):
        """
        Copy form nnUnet
        """
        super(DiceLoss, self).__init__()

        self.do_bg = do_bg
        self.class_weight = class_weight
        self.smooth = smooth

    def forward(self, x, y, batch_weight, loss_mask=None):
        assert len(self.class_weight) == x.shape[1]
        assert isinstance(x, torch.Tensor)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).cuda(x.device.index).long()
        class_weight = torch.tensor(self.class_weight).cuda(x.device.index).float()
        batch_weight = torch.tensor(batch_weight).cuda(x.device.index).float()

        shp_x = x.shape
        axes = list(range(2, len(shp_x)))  # (2, 3, 4)

        # softmax
        x = F.softmax(x, dim=1)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, False)  # (N, C+1)

        tp = (tp * class_weight).sum(1)
        fp = (fp * class_weight).sum(1)
        fn = (fn * class_weight).sum(1)

        tp = tp * batch_weight  # (N)
        fp = fp * batch_weight
        fn = fn * batch_weight

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / denominator
        dc = dc.mean()

        return -dc


if __name__ == "__main__":
    batch_size = 2
    class_num = 3
    loss_func = DiceLoss([1.0, 2.0, 3.0])
    ct = torch.randn((batch_size, class_num, 5, 5)).cuda()
    seg = torch.randint(0, class_num, (batch_size, 5, 5)).cuda()
    # print(ct.numpy())
    # print(seg.numpy())

    # pytorch
    print(loss_func(ct, seg, [1] * 2))
