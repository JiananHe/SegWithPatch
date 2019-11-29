import torch
import numpy as np
from skimage import measure
import random
import os

# 器官属性
organs_properties = {'organs_name': ['spleen', 'rkidny', 'lkidney', 'gallbladder', 'esophagus', 'liver', 'stomach',
                                     'aorta', 'vena', 'vein', 'pancreas', 'rgland', 'lgland'],
                     'organs_size': {1: 41254, 2: 21974, 3: 21790, 4: 3814, 5: 2182, 6: 236843, 7: 61189, 8: 13355,
                                     9: 11960, 10: 4672, 11: 11266, 12: 595, 13: 724},
                     'organs_weight': [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                     'num_organ': 13,
                     'sample_path': r'D:\Projects\OrgansSegment\SegWithPatch\samples\Training'}
# organs_properties = {'organs_index': [1, 3, 4, 5, 6, 7, 11, 14],
#                      'organs_name': ['spleen', 'left kidney', 'gallbladder', 'esophagus',
#                                      'liver', 'stomach', 'pancreas', 'duodenum'],
#                      'organs_size': {1: 33969.37777777778, 3: 21083.43820224719, 4: 3348.8214285714284,
#                                      5: 1916.685393258427, 6: 208806.8777777778, 7: 50836.01111111111,
#                                      11: 9410.111111111111, 14: 11118.544444444444},
#                      'num_organ': 8}

organs_name = organs_properties['organs_name']
num_organ = organs_properties['num_organ']
organs_size = organs_properties['organs_size']


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    a = torch.randint(1, 10, (6, 9, 48, 256, 256))
    b = torch.randint(1, 10, (6, 48, 256, 256))

