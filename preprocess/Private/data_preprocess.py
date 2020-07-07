from utils import *


dicom_root = r'C:\Users\13249\Desktop\case8-12\case8-12'
sub_dirs = ['Li Jian Guo/6', 'Xu Sheng Lan/6', 'Zhang Qi Shen/304', 'Zhou Yang/304', 'Zhuang Xiao Qin/10']
saved_dir = r'C:\Users\13249\Desktop\20200115-20200205\OrganSegmentation\PrivateData'

median_spacing = [0.76, 0.76, 3.0]
clip_min_intensity = -958
clip_max_intensity = 327
mean = 82.92
std_variance = 136.97

for case in sub_dirs:
    path = os.path.join(dicom_root, case)
    vol = read_dicom(path)
    image_array = sitk.GetArrayFromImage(vol)
    # image_array[:, :, :] = image_array[:, ::-1, :]

    # preprocessed
    old_spacing = vol.GetSpacing()
    # image_array = np.clip(image_array, clip_min_intensity, clip_max_intensity)
    # image_array = (image_array - mean) / std_variance
    # resampled_img = image_resample(image_array, old_spacing, median_spacing, 3, True)

    saved_vol = sitk.GetImageFromArray(image_array)
    saved_vol.SetDirection(vol.GetDirection())
    saved_vol.SetSpacing(old_spacing)
    saved_vol.SetOrigin(vol.GetOrigin())
    sitk.WriteImage(saved_vol, os.path.join(saved_dir, case.split('/')[0] + ".nii.gz"))
