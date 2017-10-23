from __future__ import print_function
import argparse
import imp
import random
import numpy as np
import itertools
import os
import time
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

def main(config_module, N_SEED):
    n_permutation = config_module.n_permutations
    precomputed_kernel_files = config_module.precomputed_kernel_files[0]
    cv_n_folds = config_module.cv_n_folds
    experiment_name = config_module.experiment_name

    random.seed(N_SEED)
    np.random.seed(N_SEED)

    print("Loading kernels...")
    file_npz = np.load("./kernels/" + precomputed_kernel_files)
    y = file_npz['labels']
    precomputed_kernel = file_npz['kernel']

    permutation_test_bac = np.zeros((n_permutation,))

    f = open("./results/"+experiment_name+"/single/best_clf.pkl", 'rb')
    permutation_classifiers = pickle.load(f)
    f.close()
    f = open("./results/"+experiment_name+"/single/final_BAC.pkl", 'rb')
    best_BAC = pickle.load(f)
    f.close()


    print("Starting permutation with ", n_permutation, " iterations")
    for p in range(n_permutation):

        np.random.seed(N_SEED + p)
        permuted_labels = np.random.permutation(y)
        skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=N_SEED)

        cv_test_bac = np.zeros((cv_n_folds,))

        for i, (train_index, test_index) in enumerate(skf.split(permuted_labels, permuted_labels)):
            best_svc = permutation_classifiers[i]

            y_train, y_test = permuted_labels[train_index], y[test_index]
            x_train, x_test = precomputed_kernel[train_index, :][:, train_index], precomputed_kernel[test_index, :][:,
                                                                                  train_index]
            best_svc.fit(x_train, y_train)

            y_predicted = best_svc.predict(x_test)

            cm = confusion_matrix(y_test, y_predicted)
            test_bac = np.sum(np.true_divide(np.diagonal(cm), np.sum(cm, axis=1))) / cm.shape[1]

            cv_test_bac[i] = test_bac
        permutation_test_bac[p] = cv_test_bac.mean()
        print("Permutation: ", p, " BAC: ", cv_test_bac.mean())
    print("")
    print("P-VALUE", (np.sum((permutation_test_bac>best_BAC).astype('int'))+1.)/(n_permutation+1.))


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
