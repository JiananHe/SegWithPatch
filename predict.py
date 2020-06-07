from val import *
from utils import *


def dataset_prediction(net, dataset_folder, format):
    assert format == 'dcm' or format == 'nii'
    dataset_info = json.load(open(dataset_info_file, 'r'))
    median_spacing = dataset_info['median_spacing']
    clip_min_intensity = dataset_info['clip_min_intensity']
    clip_max_intensity = dataset_info['clip_max_intensity']
    mean = dataset_info['mean']
    std_variance = dataset_info['std_variance']

    for file in os.listdir(dataset_folder):
        print("predict ", file)
        if format == "nii":
            image_vol = read_nii(os.path.join(dataset_folder, file))
        elif format == "dcm":
            image_vol = read_dicom(os.path.join(dataset_folder, file))
        else:
            raise NameError("unsupported format, only support dicom and nifit") from Exception

        # resample
        old_spacing = image_vol.GetSpacing()
        image_array = sitk.GetArrayFromImage(image_vol)
        image_array = np.clip(image_array, clip_min_intensity, clip_max_intensity)
        image_array = (image_array - mean) / std_variance
        resampled_img = image_resample(image_array, old_spacing, median_spacing, 3, True)

        predict = volume_predict(net, resampled_img)
        predict = post_process(predict)
        save_seg(predict, median_spacing, file, old_spacing, image_array.shape,
                 vol_direction=image_vol.GetDirection(), vol_origin=image_vol.GetOrigin())


if __name__ == "__main__":
    net = get_net(1)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load("./module/td_unet66-0.151-0.357.pth"))
    net.eval()

    dataset_prediction(net, "../PrivateData/", "dcm")
