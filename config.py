__author__ = 'mizeshuang'

import os

class project:

    project_path = '.'


    casia_dataset_b_path = 'D:\\data\\GaitDatasetB-silh\\DatasetB\\silhouettes'

    GEI_test_path = ''

    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)
    
    joint_feature_dir = '.\\joint-feature'
    cache_dir = '%s\\data\\cache\\' % project_path

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)