import SimpleITK as sitk
import numpy as np
import cv2


angle_interval = 10  # angle_interval，一共(360/angle_interval)*(180/angle_interval-2)+2条射线
line_points = 200  # 每条射线上取样500个点


def range_to_box(range_values):
    """

    :param range_values: (n, 2, 3)
    :return: (n, 8, 3)
    """
    n = len(range_values)
    box_points = np.zeros((n, 8, 3))
    a = np.tile(np.repeat(range_values[:, :, 1], 2), 2)
    box_points[:, :, 0] = np.repeat(range_values[:, :, 0], 4, axis=1)
    box_points[:, :, 1] = np.tile(np.repeat(range_values[:, :, 1], 2, axis=1), 2).reshape(n, 8)
    box_points[:, :, 2] = np.tile(range_values[:, :, 2], 4).reshape(n, 8)
    return box_points


def interpolate3d(boxes_value, samples_points, spacing, mode="nearest"):
    """

    :param boxes_value: (n, 8)
    :param samples_points: (n ,3), float
    :return: (n, 1)
    """
    num_lines = len(boxes_value)
    dis_ratio = samples_points - np.floor(samples_points)
    dis_to_floor = dis_ratio * spacing
    dis_to_ceil = (1-dis_ratio) * spacing
    dis_to_bound = np.array(np.concatenate((dis_to_floor, dis_to_ceil), axis=1)).reshape(num_lines, 2, 3)
    dises_to_box_points = range_to_box(dis_to_bound)  # (n, 8, 3)
    dis_to_box_points = np.sqrt(np.sum(dises_to_box_points**2, axis=-1)).squeeze()  # (n, 8)

    if mode == "nearest":
        targets_value = boxes_value[np.arange(num_lines), np.argmin(dis_to_box_points, axis=-1)]  # (n,)
    elif mode == "trilinear":
        targets_value = np.sum(boxes_value * dises_to_box_points, axis=-1) / np.sum(dises_to_box_points, axis=1)  # (n,)

    return targets_value


def algorithm(img, lbl, spacing, mode='label'):
    shape = img.shape
    vol_bound = np.array(shape) - 1
    assert img.shape == lbl.shape
    print(spacing)
    print(shape)

    # spleen
    organ = (lbl == 1) + 0
    if mode == 'label':
        vol_array = organ
    elif mode == 'image':
        vol_array = img / np.max(img)
    organ_points = np.array(np.where(organ))  # (3, n)
    # centre point
    organ_centre = np.mean(organ_points, axis=1).astype(np.int)
    # bounding points
    coord_bound = np.array([np.min(organ_points, axis=1), np.max(organ_points, axis=1)])  # (2, 3)
    bound_points = range_to_box(np.expand_dims(coord_bound, 0)).squeeze()
    #
    # bound_points = np.zeros((8, 3))
    # bound_points[:, 0] = np.repeat(coord_bound[:, 0], 4)
    # bound_points[:, 1] = np.tile(np.repeat(coord_bound[:, 1], 2), 2)
    # bound_points[:, 2] = np.tile(coord_bound[:, 2], 4)

    # 计算器官中心点到器官边界点的距离的最大值，以及器官中心点到volume边界的最小值
    line_length1 = np.max(np.sqrt(np.sum(((bound_points-organ_centre)*spacing)**2, axis=1)))
    line_length2 = np.sqrt(np.sum((-organ_centre*spacing)**2))
    line_length3 = np.sqrt(np.sum(((vol_bound-organ_centre)*spacing)**2))
    length_interval = np.min([line_length1, line_length2, line_length3]) / (line_points - 1)

    # 以organ_centre为球坐标原点，射线在X0Y平面的投影与X正轴夹角为alpha，射线与Z正轴夹角为beta
    all_alpha_beta = [[0, 0]]
    all_alpha_beta += [[i, j] for j in range(angle_interval, 180, angle_interval) for i in range(0, 360, angle_interval)]
    all_alpha_beta += [[0, 180]]
    all_alpha_beta = np.array(all_alpha_beta).astype(np.float) * (np.pi / 180)
    num_lines = len(all_alpha_beta)

    sample_points_value = np.zeros((line_points, num_lines))
    sample_points_value[0, :] = vol_array[tuple(organ_centre)]
    centre_physical_coord = organ_centre * spacing
    for i in range(1, line_points):
        # 获取n条射线当前取样点的物理坐标
        print(i)
        dis_to_centre = i * length_interval
        relative_physical_coord = np.array([dis_to_centre * np.cos(all_alpha_beta[:, 1]),
                                            dis_to_centre * np.sin(all_alpha_beta[:, 1]) * np.cos(all_alpha_beta[:, 0]),
                                            dis_to_centre * np.sin(all_alpha_beta[:, 1]) * np.sin(all_alpha_beta[:, 0])]).swapaxes(0, 1)
        samples_physical_coord = relative_physical_coord + centre_physical_coord  # (n, 3)
        # 获取n条射线当前取样点的8领域点的图像坐标
        samples_points = samples_physical_coord / spacing
        sample_points_range = np.array(np.concatenate((np.ceil(samples_points), np.floor(samples_points)), axis=1)).reshape(num_lines, 2, 3)
        sample_points_box = range_to_box(sample_points_range)  # (n, 8, 3)
        sample_points_box = sample_points_box.reshape(num_lines*8, 3).astype(np.int)  # (n*8， 3）

        # 处理可能的越界
        sample_points_box[:, 0][sample_points_box[:, 0] >= vol_bound[0]-1] = vol_bound[0]-1
        sample_points_box[:, 1][sample_points_box[:, 1] >= vol_bound[1]-1] = vol_bound[1]-1
        sample_points_box[:, 2][sample_points_box[:, 2] >= vol_bound[2]-1] = vol_bound[2]-1

        try:
            sample_boxes_values = np.array([vol_array[tuple(p)] for p in sample_points_box]).reshape(num_lines, 8)
        except:
            mins = np.min(sample_points_box, axis=0)
            maxs = np.max(sample_points_box, axis=0)
            print("out of range")
        targets_value = interpolate3d(sample_boxes_values, samples_points, spacing)  # (n,)
        sample_points_value[i, :] = targets_value

    print(sample_points_value.shape)
    # 放缩0-1
    sample_points_value = (sample_points_value - np.min(sample_points_value)) \
                          / (np.max(sample_points_value) - np.min(sample_points_value))
    # cv2.imshow(mode, sample_points_value[:, 1:1+36*4])
    cv2.imshow(mode, sample_points_value)

    save_name = img_path if mode == "image" else lbl_path
    save_name = save_name.split("\\")[-1].split(".")[0]
    sample_points_value *= 255
    cv2.imwrite("%s.jpg" % save_name, sample_points_value)

if __name__ == "__main__":
    img_path = r"D:\Projects\OrgansSegment\BTCV\RawData\Training\img\img0001.nii.gz"
    lbl_path = r"D:\Projects\OrgansSegment\BTCV\RawData\Training\label\label0001.nii.gz"
    img_vol = sitk.ReadImage(img_path)
    lbl_vol = sitk.ReadImage(lbl_path)

    img = sitk.GetArrayFromImage(img_vol)
    lbl = sitk.GetArrayFromImage(lbl_vol)

    spacing = img_vol.GetSpacing()
    spacing = np.array([spacing[2], spacing[0], spacing[1]])

    # lbl = np.zeros((120, 512, 512))
    # lbl[40:100, 100:350, 120:400] = 1
    # img = np.zeros((120, 512, 512))
    # img[40:100, 100:350, 120:400] = 1
    # spacing = np.array([3, 0.5, 0.5])

    algorithm(img, lbl, spacing, "label")
    algorithm(img, lbl, spacing, "image")
    cv2.waitKey(0)


