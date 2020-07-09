import os
import SimpleITK as sitk
import torch
import numpy as np
import csv
import json
from utils import organs_properties, calc_weight_matrix
from models.td_unet import get_net
# from models.td_unet_cnn import get_net
# from models.dense_vnet import get_net
import scipy.ndimage as ndimage
from utils import *


patch_weight_matrix = calc_weight_matrix(patch_size)

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
        prediction = outputs[-1].squeeze().cpu().detach().numpy()  # (bs, cls, *patch_size)

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
                    prediction = patch_predict(net, patch_batch)  # (bs, cls, *patch_size)
                    for i in range(val_batch_size):
                        vol_predict[:,
                        patch_border[i][0]:patch_border[i][1],
                        patch_border[i][2]:patch_border[i][3],
                        patch_border[i][4]:patch_border[i][5]] += (prediction[i] * patch_weight_matrix)
                    sample_count = 0

    vol_predict = np.argmax(vol_predict, axis=0).astype(np.uint8)
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


def post_process(input):
    """
    分割结果的后处理：保留最大连通区域
    :param input: whole volume （D, W, H）
    :return: （D, W, H）
    """
    s = input.shape
    output = np.zeros((num_organ + 1, s[0], s[1], s[2]))
    for id in range(1, num_organ + 1):
        org_seg = (input == id) + .0
        if (org_seg == .0).all():
            continue
        labels, num = measure.label(org_seg, return_num=True)
        regions = measure.regionprops(labels)
        regions_area = [regions[i].area for i in range(num)]

        # omit_region_id = []
        # for rid, area in enumerate(regions_area):
        #     if area < 0.1 * organs_size[organs_index[id - 1]]:  # 记录区域面积小于10%*器官尺寸的区域的id
        #         omit_region_id.append(rid)
        # for idx in omit_region_id:
        #     org_seg[labels == (idx+1)] = 0

        region_num = regions_area.index(max(regions_area)) + 1  # 记录面积最大的区域，不会计算background(0)
        org_seg[labels == region_num] = 1
        org_seg[labels != region_num] = 0

        output[id, :, :, :] = org_seg

    return np.argmax(output, axis=0)


def save_seg(predict_array, predict_spacing, ct_name, raw_spacing, shape_before_resample, shape_before_crop=None, crop_coords=None, **kwargs):
    # 需要放缩保存为原spacing
    if predict_spacing != raw_spacing:
        predict_array = image_resize(predict_array, shape_before_resample, 0, True)
    assert predict_array.shape == shape_before_resample

    # 若有crop则还原
    if shape_before_crop is not None:
        assert crop_coords is not None
        saved_array = np.zeros(shape_before_crop, dtype=np.uint8)
        saved_array[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], crop_coords[4]:crop_coords[5]] = predict_array
    else:
        saved_array = predict_array

    # 将raw_ct的prediction保存
    pred_vol = sitk.GetImageFromArray(saved_array)
    pred_vol.SetSpacing(raw_spacing)
    if kwargs["vol_direction"]:
        pred_vol.SetDirection(kwargs["vol_direction"])
    if kwargs["vol_origin"]:
        pred_vol.SetOrigin(kwargs["vol_origin"])

    saved_name = ct_name if len(ct_name) > 6 and ct_name[-7:] == '.nii.gz' else ct_name + '.nii.gz'
    sitk.WriteImage(pred_vol, os.path.join('./prediction', saved_name))


def dataset_validation(net, val_samples_info, crop_roi=False, cal_acc=True, show_sample_dice=False, save=False, postprocess=False):
    sample_dices = []

    if save:
        median_spacing = json.load(open(dataset_info_file, 'r'))['median_spacing']
    for val_info in val_samples_info:
        img = np.load(val_info[0])
        seg = None
        name = val_info[0].split("/")[-1][:-4]

        if crop_roi:
            seg = np.load(val_info[1])
            z_min = np.min(np.where(seg != 0)[0])
            z_max = np.max(np.where(seg != 0)[0]) + 1
            x_min = np.min(np.where(seg != 0)[1])
            x_max = np.max(np.where(seg != 0)[1]) + 1
            y_min = np.min(np.where(seg != 0)[2])
            y_max = np.max(np.where(seg != 0)[2]) + 1

            img = img[z_min:z_max, x_min:x_max, y_min:y_max]
            seg = seg[z_min:z_max, x_min:x_max, y_min:y_max]

        predict = volume_predict(net, img)
        assert predict.shape == img.shape

        if postprocess:
            predict = post_process(predict)

        if cal_acc:
            if seg is None:
                seg = np.load(val_info[1])
            dices = volume_accuarcy(predict, seg)
            sample_dices.append(dices)

            if show_sample_dice:
                os.system('echo %s' % name)
                s = " ".join(["%s:%s" % (i, j if j == 'None' else round(j, 3)) for i, j in zip(classes_name, dices)])
                os.system('echo %s' % s)

        if save:
            save_seg(predict, median_spacing, name, eval(val_info[2]), eval(val_info[-2]), eval(val_info[-3]), eval(val_info[-1]))

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
    net = get_net(1)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load("./module/td_unet66-0.151-0.357.pth"))
    net.eval()

    _, _, val_samples_info, val_samples_name = split_train_val()
    dataset_validation(net, val_samples_info, cal_acc=True, show_sample_dice=True, save=True, postprocess=True)
