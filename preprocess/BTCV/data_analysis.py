import SimpleITK as sitk
import os
import numpy as np

# 记录各器官的体积
orgs_id = list(range(1, 14))
orgs_size = {i: .0 for i in orgs_id}
orgs_count = {i: 0 for i in orgs_id}

# 分析BTCV训练数据
btcv_path = r'D:\Projects\OrgansSegment\BTCV\RawData\Training'
btcv_img_path = os.path.join(btcv_path, 'img')
btcv_lbl_path = os.path.join(btcv_path, 'label')
for case in os.listdir(btcv_lbl_path):
    # read label
    vol = sitk.ReadImage(os.path.join(btcv_lbl_path, case))
    array = sitk.GetArrayFromImage(vol)
    print(case, vol.GetSpacing(), array.shape, array.dtype)

    for id in orgs_id:
        if np.any((array==id)):
            orgs_size[id] += np.sum(array==id)
            orgs_count[id] += 1

    # 开始结束slice
    slices = np.any(array, axis=(1, 2))  # len(slices) = depth of volume
    start_slice, end_slice = np.where(slices)[0][[0, -1]]
    print('start: %d, end: %d' % (start_slice, end_slice))

    # read image
    case_name = case.replace('label', 'img')
    vol1 = sitk.ReadImage(os.path.join(btcv_img_path, case_name))
    array1 = sitk.GetArrayFromImage(vol1)
    # assert vol.GetSpacing()[-1] == vol1.GetSpacing()[-1] and array.shape[1:] == array1.shape[1:]
    print(case_name, vol1.GetSpacing(), array1.shape, array1.dtype)

    print('-----------------------\n')

print(orgs_count)
# {1: 30, 2: 30, 3: 30, 4: 28, 5: 30, 6: 30, 7: 30, 8: 30, 9: 30, 10: 30, 11: 30, 12: 30, 13: 30}

orgs_size_avg = {i:orgs_size[i]/orgs_count[i] for i in orgs_id if orgs_count[i] != 0}
print(orgs_size_avg)
# {1: 150205.6, 2: 73771.76666666666, 3: 74399.06666666667, 4: 12981.17857142857, 5: 7322.133333333333,
# 6: 798498.6666666666, 7: 203866.7, 8: 45466.4, 9: 41663.6, 10: 16301.066666666668, 11: 38546.666666666664,
# 12: 1973.9333333333334, 13: 2453.6}

max_org = max(list(orgs_size_avg.values()))
orgs_size_ratio = {i:max_org/orgs_size_avg[i] for i in orgs_size_avg.keys()}
print(orgs_size_ratio)
# {1: 5.316037928457172, 2: 10.823905983906219, 3: 10.732643599471677, 4: 61.51203161353571, 5: 109.05273508631365,
# 6: 1.0, 7: 3.9167684897370028, 8: 17.562390395251583, 9: 19.165378571862888, 10: 48.98444273585368,
# 11: 20.71511587685922, 12: 404.52159816272075, 13: 325.4396261275948}

print("############################################")
# 分析BTCV测试数据
btcv_path = r'D:\Projects\OrgansSegment\BTCV\RawData\Testing'
btcv_img_path = os.path.join(btcv_path, 'img')
for case in os.listdir(btcv_img_path):
    # read image
    vol1 = sitk.ReadImage(os.path.join(btcv_img_path, case))
    array1 = sitk.GetArrayFromImage(vol1)
    # assert vol.GetSpacing()[-1] == vol1.GetSpacing()[-1] and array.shape[1:] == array1.shape[1:]
    print(case, vol1.GetSpacing(), array1.shape, array1.dtype)
    print('-----------------------\n')
