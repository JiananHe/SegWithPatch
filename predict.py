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
from val import dataset_prediction
from utils import read_dicom
from preprocess.BTCV.data_preprocess import preprocess


if __name__ == "__main__":
    # # preprocess
    # info = []
    # info_file = open('./csv_files/private_data_info.csv', 'w', newline="")
    # raw_img_path = r"D:\Projects\OrgansSegment\Data\PrivateData\RawData"
    # sample_img_path = r"D:\Projects\OrgansSegment\Data\PrivateData\Samples"
    # info_writer = csv.writer(info_file)
    #
    # for case in os.listdir(raw_img_path):
    #     ct_vol = read_dicom(os.path.join(raw_img_path, case, os.listdir(os.path.join(raw_img_path, case))[0]))
    #     # save as nii
    #     sitk.WriteImage(ct_vol, r"D:\Projects\OrgansSegment\Data\PrivateData\RawDataNii\img\%s.nii.gz" % case)
    #
    #     preprocess(case, ct_vol, sample_img_path, info)
    #
    # for f in info:
    #     info_writer.writerow(f)
    # info_writer.close()

    # prediction
    # models
    net = get_net(False)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load('./module/td_unet410-0.373.pth'))
    net.eval()

    dataset_prediction(net, 'csv_files/private_data_info.csv',
                       raw_data_dir=r'D:\Projects\OrgansSegment\Data\PrivateData\RawDataNii', cal_acc=False, save=True, postprocess=True)