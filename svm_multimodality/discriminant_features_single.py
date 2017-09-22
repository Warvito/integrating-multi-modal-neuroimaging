from __future__ import print_function
import argparse
import imp
import random
import numpy as np
import pickle
import glob
import time
import re
import nibabel as nib
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    return sorted(l, key=alphanum_key)

def lower_tri_witout_diag(x):
    return np.concatenate([ x[i][:i] for i in xrange(x.shape[0])])


def main(config_module, N_SEED):
    data_dirs = config_module.data_dirs
    type_data = config_module.type_data
    precomputed_kernel_files = config_module.precomputed_kernel_files
    experiment_name = config_module.experiment_name

    cv_n_folds = config_module.cv_n_folds

    random.seed(N_SEED)
    np.random.seed(N_SEED)

    file_npz = np.load("./kernels/" + precomputed_kernel_files[0])
    y = file_npz['labels']

    f = open("./results/"+experiment_name+"/single/best_clf.pkl", 'rb')
    saved_classifiers = pickle.load(f)
    f.close()

    for m in range(len(data_dirs)):
        if type_data[m] == "MATRIX":
            input_data_type = ".txt"
        if type_data[m] == "3D":
            input_data_type = ".nii"

        print("DATA #"+str(m)+" Data type "+type_data[m])
        print("Loading")
        paths_train = glob.glob(data_dirs[m] + "/*" + input_data_type)
        paths_train = sort_nicely(paths_train)

        dataset = []
        for k, path in enumerate(paths_train):
            if type_data[m] == "MATRIX":
                matrix = np.genfromtxt(path)
                matrix = lower_tri_witout_diag(matrix)
                matrix = np.asarray(matrix, dtype='float64')
                img_vec = np.reshape(matrix, np.product(matrix.shape))
                dataset.append(img_vec)
            if type_data[m] == "3D":
                VOLUME = nib.load(path)
                img = VOLUME.get_data()
                img_dims = img.shape
                img = np.asarray(img, dtype='float64')
                img_vec = np.reshape(img, np.product(img.shape))
                dataset.append(img_vec)

        dataset = np.asarray(dataset)
        print("DATA SHAPE")
        print(dataset.shape)

        skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=N_SEED)

        coefs_ = []
        print("Calculating for each fold of the cross validation")
        for i, (train_index, test_index) in enumerate(skf.split(y, y)):
            start_time = time.time()
            best_classifiers = saved_classifiers[i]
            y_train, y_test = y[train_index], y[test_index]
            x_train, x_test = dataset[train_index, :], dataset[test_index, :]

            svc = SVC(kernel='linear', C = best_classifiers.C,  probability=True, random_state=1)
            svc.fit(x_train, y_train)
            weights = svc.coef_
            if type_data[m] == "MATRIX":
                C = np.zeros((90, 90))
                l = 0
                for a in range(1, 90):
                    for b in range(a):
                        C[a][b] = weights[0, l]
                        l = l + 1
                C = C + C.T
                C = np.abs(C)
                coefs_.append(C)
            if type_data[m] == "3D":
                img = np.reshape(weights, img_dims)
                coefs_.append(img)


            stop_time = time.time()
            print("ETA: ", (i-cv_n_folds)*(stop_time - start_time), " seconds")

        coefs_ = np.asarray(coefs_)
        coefs_ = np.mean(coefs_, axis=0)
        print("")
        print("SHAPE OF SVM WEIGHTS")
        print(coefs_.shape)
        print("SAVING SVM WEIGHTS IN " + "./results/" + experiment_name + "/single/SVM_LEARNED_FEATURES" + str(m))
        if type_data[m] == "MATRIX":
            np.savetxt("./results/"+experiment_name+"/single/SVM_LEARNED_FEATURES"+str(m)+".csv", coefs_, delimiter=',')
        if type_data[m] == "3D":
            empty_header = nib.Nifti1Header()
            another_img = nib.Nifti1Image(coefs_, VOLUME.affine, empty_header)
            nib.save(another_img,"./results/"+experiment_name+"/single/SVM_LEARNED_FEATURES"+str(m)+".nii")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., Combined")
    args = parser.parse_args()
    config_name = args.config_name

    try:
        config_module = imp.load_source('config', config_name)

    except IOError:
        print('Cannot open ',config_name, '. Please specify the correct path of the configuration file. Example: python general_AV_SVM.py ./config/config_test.py')

    if np.isscalar(config_module.N_SEED):
        main(config_module, config_module.N_SEED)
    else:
        print("Please report the N_seed as a number.")
