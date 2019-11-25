import os
import SimpleITK as sitk
import torch
import numpy as np
import csv
from utils import organs_properties
from models.vnet import get_net
# from models.dense_vnet import get_net
import scipy.ndimage as ndimage

organs_name = organs_properties['organs_name']
sample_path = organs_properties['sample_path']
num_organ = organs_properties['num_organ']

model_dir = './module/vnet0-0.022-0.025.pth'

sample_size = 64  # 64*64*64 for a patch


def volume_predict(net, vol_array):
    """
    the accuracy of a volume
    :param output: (B, 14, 64, 64, 64)
    :param target: (B, 64, 64, 64)
    :return: prediction
    """
    net.eval()
    # 切割
    s = vol_array.shape
    patches = []
    for z in range(0, s[0], sample_size):
        z_end = s[0] if z + sample_size > s[0] else z + sample_size
        z_start = z_end - sample_size
        for x in range(0, s[1], sample_size):
            x_end = s[1] if x + sample_size > s[1] else x + sample_size
            x_start = x_end - sample_size
            for y in range(0, s[2], sample_size):
                y_end = s[2] if y + sample_size > s[2] else y + sample_size
                y_start = y_end - sample_size
                patches.append(vol_array[z_start:z_end, x_start:x_end, y_start:y_end])

    # 预测
    patch_predicts = []
    with torch.no_grad():
        for patch in patches:
            patch_tensor = torch.FloatTensor(patch).cuda()
            patch_tensor = patch_tensor.unsqueeze(dim=0)
            patch_tensor = patch_tensor.unsqueeze(dim=0)  # (1, 1, 64, 64, 64)

            output = net(patch_tensor)
            output = output.squeeze().cpu().detach().numpy()  # (14, 64, 64, 64)
            output = np.argmax(output, axis=0)  # (64, 64, 64)

            patch_predicts.append(output)

    # 拼接
    vol_predict = np.zeros(vol_array.shape)
    patch_idx = 0
    for z_start in range(0, s[0], sample_size):
        z_end = s[0] if z_start + sample_size > s[0] else z_start + sample_size
        for x_start in range(0, s[1], sample_size):
            x_end = s[1] if x_start + sample_size > s[1] else x_start + sample_size
            for y_start in range(0, s[2], sample_size):
                y_end = s[2] if y_start + sample_size > s[2] else y_start + sample_size
                vol_predict[z_start:z_end, x_start:x_end, y_start:y_end] = \
                    patch_predicts[patch_idx][0:z_end-z_start, 0:x_end-x_start, 0:y_end-y_start]
                patch_idx += 1

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


def save_seg(pred_seg, info, accs):
    prediction = np.argmax(pred_seg, axis=1).squeeze()  # (D, 256, 256)

    img_name = info[0]
    ct = sitk.ReadImage((os.path.join(image_path, img_name)))

    # 插值
    ct_array = sitk.GetArrayFromImage(ct)
    pred_unsample = ndimage.zoom(prediction, (ct_array.shape[0]/prediction.shape[0],
                                              ct_array.shape[1]/prediction.shape[1],
                                              ct_array.shape[2]/prediction.shape[2]), order=0)

    # 保存
    pred = sitk.GetImageFromArray(pred_unsample)

    pred.SetDirection(ct.GetDirection())
    pred.SetOrigin(ct.GetOrigin())
    pred.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred, os.path.join('./prediction', img_name))
    del pred
    csv_writer.writerow([img_name]+accs)


def dataset_accuracy(net, csv_path, cal_acc = True, save=False, postprocess=False):
    reader = csv.reader(open(csv_path))
    sample_dices = []
    for line in reader:
        vol_path = line[0]
        vol = sitk.ReadImage(vol_path)
        vol_array = sitk.GetArrayFromImage(vol)
        vol_predict = volume_predict(net, vol_array)

        if cal_acc:
            lbl_path = line[1]
            vol_label = sitk.ReadImage(lbl_path)
            vol_label = sitk.GetArrayFromImage(vol_label)
            dices = volume_accuarcy(vol_label, vol_predict)
            sample_dices.append(dices)

        if save:
            pass

    if cal_acc:
        sample_dices = np.array(sample_dices)
        org_mean_dice = [np.mean(np.array(list(set(sample_dices[:, i]).difference(['None'])), dtype=np.float16))
                         for i in range(len(organs_name))]
        os.system('echo %s' % "mean organs dice:")
        s = " ".join(["%s:%.3f" % (i, j) for i, j in zip(organs_name, org_mean_dice)])
        os.system('echo %s' % s)

        return org_mean_dice





if __name__ == "__main__":
    # models
    net = get_net(False)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(model_dir))
    net.eval()

    val_org_mean_dice = dataset_accuracy(net, 'csv_files/btcv_val_info.csv')
