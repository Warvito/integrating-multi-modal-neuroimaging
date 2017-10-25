from __future__ import print_function
import argparse
import imp
import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------------
# CHANGE HERE
# ---------------------------------------------------------------------------------

mask_file = "./masks/aal_MNI_V4.img"
features_file = './results/2_single_COBRE/single/SVM_LEARNED_FEATURES0.nii'
csv_save_file = './results/2_single_COBRE/single/SVM_LEARNED_FEATURES0.csv'

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


print("MASK FILE " + mask_file)
MASK = nib.load(mask_file)
img_mask = MASK.get_data()
img_mask = np.asarray(img_mask, dtype='int')
print("MASK DIM " + str(img_mask.shape))


FEATURES = nib.load(features_file)
img_feat = FEATURES.get_data()
img_feat = np.asarray(img_feat, dtype='float32')
print("FEATURES DIM " + str(img_feat.shape))

discriminant_weight = np.zeros((np.max(img_mask),1))
for i in range(1, int(np.max(img_mask))+1):
    discriminant_weight[i-1,0] = np.mean(img_feat[img_mask==i])

np.savetxt(csv_save_file, discriminant_weight)
