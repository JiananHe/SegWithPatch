import torch
from torch import nn
import numpy as np
from skimage.transform import resize
from loss.ce_loss import CELoss
from loss.dice_loss import DiceLoss
import sys
sys.path.append("..")
from utils import *


class DC_and_CE_loss(nn.Module):
    def __init__(self, class_weight, weight_ce=1, weight_dice=1):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        """
        super(DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

        self.ce = CELoss(class_weight)
        self.dc = DiceLoss(class_weight)

    def forward(self, net_outputs, target, batch_weight=None):
        if batch_weight is not None:
            assert len(batch_weight) == target.shape[0]
        else:
            batch_weight = [1.0] * target.shape[0]

        # multi-scale network output, target should be a numpy array
        assert isinstance(target, np.ndarray)
        assert net_outputs[-1].shape[-3:] == target.shape[-3:]
        # delete the channel dimension in target
        if len(target.shape) == 5 and target.shape[1] == 1:
            target = np.squeeze(target, axis=1)

        dc_losses = []
        ce_losses = []
        num_output = len(net_outputs)
        for output in net_outputs:  # from low stage to high stage
            if output.shape[-3:] != target.shape[-3:]:
                scaled_target = image_resize(target, output.shape[-3:], order=0, is_anisotropic=True)
            else:
                scaled_target = target
            dc_losses.append(self.dc(output, scaled_target, batch_weight) if self.weight_dice != 0 else 0)
            ce_losses.append(self.ce(output, scaled_target, batch_weight) if self.weight_ce != 0 else 0)

        stage_weights = [np.power(0.5, i) for i in range(num_output)]
        stage_weights.reverse()
        weight_sum = np.sum(stage_weights)
        stage_weights = [i / weight_sum for i in stage_weights]

        total_ce_loss = ce_losses[0] * stage_weights[0]
        total_dc_loss = dc_losses[0] * stage_weights[0]
        for i in range(1, num_output):
            total_ce_loss += ce_losses[i] * stage_weights[i]
            total_dc_loss += dc_losses[i] * stage_weights[i]

        total_ce_loss = total_ce_loss * self.weight_ce
        total_dc_loss = total_dc_loss * self.weight_dice
        return total_ce_loss, total_dc_loss
