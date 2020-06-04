import os
import SimpleITK as sitk
import torch
import numpy as np
import csv
from utils import organs_properties, calc_weight_matrix, post_process
from models.td_unet import get_net
# from models.td_unet_cnn import get_net
# from models.dense_vnet import get_net
import scipy.ndimage as ndimage
from utils import *


def patch_predict(net, patch):
    """
    forward to predict
    :param net: module
    :param patch: (8, 64, 64, 64)
    :param slices: (8, 6, 140, 140)
    :return:
    """
    with torch.no_grad():
        patch_tensor = torch.FloatTensor(patch).unsqueeze(1).cuda()  # (bs, 1, *patch_size)
        outputs = net(patch_tensor)
        prediction = outputs[-1].squeeze().cpu().detach().numpy()  # (cls, *patch_size)

        del patch_tensor, outputs
        torch.cuda.empty_cache()
    return prediction


def volume_predict(net, vol_array):
    net.eval()
    vs = vol_array.shape

    # 切割patch（重叠sample_size/2） -> patch predict -> prediction填充
    vol_predict = np.zeros((num_organ + 1, *vs))
    steps = compute_steps_for_sliding_window(patch_size, vs)

    patch_batch = np.zeros((val_batch_size, *patch_size))
    patch_border = np.zeros((val_batch_size, 6), dtype=np.int)
    sample_count = 0
    for z_start in steps[0]:
        z_end = z_start + patch_size[0]
        for x_start in steps[1]:
            x_end = x_start + patch_size[1]
            for y_start in steps[2]:
                y_end = y_start + patch_size[2]

                patch = vol_array[z_start:z_end, x_start:x_end, y_start:y_end]
                patch_batch[sample_count] = patch
                patch_border[sample_count] = [z_start, z_end, x_start, x_end, y_start, y_end]
                sample_count += 1

                # predict validation batch
                if sample_count == val_batch_size:
                    prediction = patch_predict(net, patch_batch)  # (14, *patch_size)
                    for i in range(val_batch_size):
                        vol_predict[:,
                        patch_border[i][0]:patch_border[i][1],
                        patch_border[i][2]:patch_border[i][3],
                        patch_border[i][4]:patch_border[i][5]] += (prediction[i] * weight_matrix)
                    sample_count = 0

    vol_predict = np.argmax(vol_predict, axis=0)
    return vol_predict


def volume_accuarcy(vol_label, vol_predict):
    """
    return dice of a volume
    :param vol_label:
    :param vol_predict:
    :return:
    """
    assert vol_predict.shape == vol_label.shape
    dices = []
    for org_idx in range(0, num_organ + 1):
        org_label = (vol_label == org_idx) + 0.
        org_predict = (vol_predict == org_idx) + 0.
        if (org_label == 0.).all() and (org_predict == 0.).all():
            dices.append('None')
            continue

        org_dice = 2 * np.sum(org_label * org_predict) / (np.sum(org_label**2) + np.sum(org_predict**2))
        dices.append(org_dice)

    return dices


def resample_to_raw_spacing(prediction, raw_ct):
    raw_ct_array = sitk.GetArrayFromImage(raw_ct)
    raw_shape = raw_ct_array.shape
    pred_shape = prediction.shape
    resample_prediction = ndimage.zoom(prediction, (raw_shape[0] / pred_shape[0],
                                                    raw_shape[1] / pred_shape[1],
                                                    raw_shape[2] / pred_shape[2]), order=0)
    return resample_prediction


def save_seg(ct_name, raw_ct, ct_predict):
    # 将raw_ct的prediction保存
    pred_vol = sitk.GetImageFromArray(ct_predict)

    pred_vol.SetDirection(raw_ct.GetDirection())
    pred_vol.SetOrigin(raw_ct.GetOrigin())
    pred_vol.SetSpacing(raw_ct.GetSpacing())

    sitk.WriteImage(pred_vol, os.path.join('./prediction', ct_name))


def dataset_validation(net, val_samples_info, cal_acc=True, show_sample_dice=False, save=False, postprocess=False):
    sample_dices = []

    for val_info in val_samples_info:
        img = np.load(val_info[0])
        predict = volume_predict(net, img)
        assert predict.shape == img.shape

        if postprocess:
            predict = post_process(predict)

        if cal_acc:
            seg = np.load(val_info[1])
            dices = volume_accuarcy(predict, seg)
            sample_dices.append(dices)

            if show_sample_dice:
                name = val_info[0].split("/")[-1]
                os.system('echo %s' % name)
                s = " ".join(["%s:%s" % (i, j if j == 'None' else round(j, 3)) for i, j in zip(classes_name, dices)])
                os.system('echo %s' % s)

        if save:
            pass

    # mean organ loss
    if cal_acc:
        sample_dices = np.array(sample_dices)
        cls_mean_dice = [np.mean(np.array(list(set(sample_dices[:, i]).difference(['None'])), dtype=np.float16))
                         for i in range(len(classes_name))]
        os.system('echo %s' % "mean organs dice:")
        s = " ".join(["%s:%.3f" % (i, j) for i, j in zip(classes_name, cls_mean_dice)])
        os.system('echo %s' % s)

        return cls_mean_dice  # (cls)
    return None


if __name__ == "__main__":
    lbl = np.random.randint(0, 14, (5, 8, 8))
    # pred = np.random.randint(0, 14, (5, 8, 8))
    print(lbl[2, 6, 5])
    pred = lbl.copy()
    pred[2, 6, 5] = 0
    print(volume_accuarcy(pred, lbl))
