from __future__ import print_function
import argparse
import imp
import nibabel as nib
import numpy as np




def main(config_module):
    type_data = config_module.type_data
    mask_data = config_module.mask_roi_file

    experiment_name = config_module.experiment_name

    for m in range(len(type_data)):
        print(m)
        print(type_data[m])
        if type_data[m] == "3D":
            print("MASK ROI " + mask_data[m])
            MASK = nib.load(mask_data[m])
            img_mask = MASK.get_data()
            img_dims = img_mask.shape
            img_mask = np.asarray(img_mask, dtype='int')
            print(np.min(img_mask))
            print(img_dims)
            sum = 0
            for i in range(1, int(np.max(img_mask))):
                sum = np.sum(img_mask[img_mask==1])
            A = img_mask[img_mask==1]
            print(A)
            print(sum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., Combined")
    args = parser.parse_args()
    config_name = args.config_name

    try:
        config_module = imp.load_source('config', config_name)

    except IOError:
        print('Cannot open ', config_name,
              '. Please specify the correct path of the configuration file. Example: python general_AV_SVM.py ./config/config_test.py')

    main(config_module)
