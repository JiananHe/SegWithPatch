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

organs_name = organs_properties['organs_name']
sample_path = organs_properties['sample_path']
num_organ = organs_properties['num_organ']

model_dir = './module/td_unet410-0.373.pth'

sample_size = 64  # 64*64*64 for a patch
slice_size = 140  # 140*140 for a slice
sample_stride = int(sample_size / 2)
batch_size = 4


def patch_predict(net, patch):
    """
    forward to predict
    :param net: module
    :param patch: (8, 64, 64, 64)
    :param slices: (8, 6, 140, 140)
    :return:
    """
    with torch.no_grad():
        patch_tensor = torch.FloatTensor(patch).unsqueeze(1).cuda()  # (8, 1, 64, 64, 64)

        output = net(patch_tensor)
        prediction = output.squeeze().cpu().detach().numpy()  # (14, 64, 64, 64)
        del patch_tensor, output
        torch.cuda.empty_cache()

    return prediction


def patch_border(start, s):
    end = start + sample_size
    if end >= s:
        return s - sample_size, s, 1
    else:
        return start, end, 0


def volume_predict(net, vol_array):
    """
    the accuracy of a volume
    :param output: (B, 14, 64, 64, 64)
    :param target: (B, 64, 64, 64)
    :return: prediction
    """
    net.eval()
    vs = vol_array.shape

    # 切割patch（重叠sample_size/2） -> patch predict -> prediction填充
    weight_matrix = calc_weight_matrix(sample_size)  # （64, 64, 64)
    vol_predict = np.zeros((num_organ+1, vs[0], vs[1], vs[2]))

    patch_batch = np.zeros((batch_size, sample_size, sample_size, sample_size))
    slices_batch = np.zeros((batch_size, 6, slice_size, slice_size))
    bd = np.zeros((batch_size, 6)).astype(np.int)  # record border of batch
    sample_count = 0
    for z_start in range(0, vs[0], sample_stride):
        z_start, z_end, z_over = patch_border(z_start, vs[0])
        for x_start in range(0, vs[1], sample_stride):
            x_start, x_end, x_over = patch_border(x_start, vs[1])
            for y_start in range(0, vs[2], sample_stride):
                y_start, y_end, y_over = patch_border(y_start, vs[2])

                patch = vol_array[z_start:z_end, x_start:x_end, y_start:y_end]  # (64, 64, 64)
                patch_batch[sample_count] = patch
                bd[sample_count] = [z_start, z_end, x_start, x_end, y_start, y_end]
                sample_count += 1

                if sample_count == batch_size:
                    prediction = patch_predict(net, patch_batch)  # (14, 64, 64, 64)
                    for i in range(batch_size):
                        vol_predict[:, bd[i][0]:bd[i][1], bd[i][2]:bd[i][3], bd[i][4]:bd[i][5]] += \
                            (prediction[i] * weight_matrix)
                    sample_count = 0

                if y_over:
                    break
            if x_over:
                break
        if z_over:
            break

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
    for org_idx in range(1, num_organ + 1):
        org_label = (vol_label == org_idx) + 0.
        if (org_label == 0.).all():
            dices.append('None')
            continue
        org_predict = (vol_predict == org_idx) + 0.

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


def dataset_prediction(net, csv_path, raw_data_dir=None, cal_acc=True, show_sample_dice=False, save=False, postprocess=False):
    reader = csv.reader(open(csv_path))
    sample_dices = []
    if raw_data_dir is not None:
        raw_ct_path = os.path.join(raw_data_dir, 'img')
        raw_lbl_path = os.path.join(raw_data_dir, 'label')

    for line in reader:
        ct_path = line[0]
        ct_name = ct_path.split("\\")[-1]
        ct = sitk.ReadImage(ct_path)
        ct_array = sitk.GetArrayFromImage(ct)
        ct_predict = volume_predict(net, ct_array)
        assert ct_predict.shape == ct_array.shape

        if raw_data_dir is not None:
            # 将prediction从采用为原始分辨率
            ct = sitk.ReadImage(os.path.join(raw_ct_path, ct_name))
            ct_predict = resample_to_raw_spacing(ct_predict, ct)

        if postprocess:
            ct_predict = post_process(ct_predict)

        if cal_acc:
            if raw_data_dir is not None:
                lbl_path = os.path.join(raw_lbl_path, ct_name.replace('img', 'label'))
            else:
                lbl_path = line[1]
            vol_label = sitk.ReadImage(lbl_path)
            vol_label = sitk.GetArrayFromImage(vol_label)
            dices = volume_accuarcy(vol_label, ct_predict)
            sample_dices.append(dices)
            if show_sample_dice:
                os.system('echo %s' % ct_name)
                s = " ".join(["%s:%s" % (i, j if j == 'None' else round(j, 3)) for i, j in zip(organs_name, dices)])
                os.system('echo %s' % s)

        if save:
            save_seg(ct_name, ct, ct_predict)

    if cal_acc:
        sample_dices = np.array(sample_dices)
        org_mean_dice = [np.mean(np.array(list(set(sample_dices[:, i]).difference(['None'])), dtype=np.float16))
                         for i in range(len(organs_name))]
        os.system('echo %s' % "mean organs dice:")
        s = " ".join(["%s:%.3f" % (i, j) for i, j in zip(organs_name, org_mean_dice)])
        os.system('echo %s' % s)

        return org_mean_dice
    return None


if __name__ == "__main__":
    # models
    net = get_net(False)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(model_dir))
    net.eval()

    # val_org_mean_dice = dataset_prediction(net, 'info_files/btcv_val_info.csv', show_sample_dice=True, save=True, postprocess=True)
    val_org_mean_dice = dataset_prediction(net, 'info_files/btcv_val_info.csv',
                                           raw_data_dir=r'D:\Projects\OrgansSegment\BTCV\RawData\Training', show_sample_dice=True, save=True, postprocess=True)
    print("mean dice: %.3f" % np.mean(val_org_mean_dice))

    # val_org_mean_dice = dataset_prediction(net, 'info_files/btcv_val_info.csv',
    #                                        show_sample_dice=True, save=True, postprocess=True)
    # print("mean dice: %.3f" % np.mean(val_org_mean_dice))

    # test set
    # dataset_prediction(net, 'info_files/btcv_test_info.csv',
    #                    raw_data_dir=r'D:\Projects\OrgansSegment\BTCV\RawData\Testing', cal_acc=False, save=True, postprocess=True)

