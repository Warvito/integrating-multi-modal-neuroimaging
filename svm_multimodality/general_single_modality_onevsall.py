from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import imp
import random
import pickle
import argparse
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, confusion_matrix


def balanced_accuracy_score(actual, prediction):
    cm = confusion_matrix(actual, prediction)
    bac = np.sum(np.true_divide(np.diagonal(cm), np.sum(cm, axis=1))) / cm.shape[1]
    return bac


def main(config_module, N_SEED):

    C_values = config_module.C_values
    cv_n_folds = config_module.cv_n_folds
    precomputed_kernel_files = config_module.precomputed_kernel_files[0]
    n_jobs = config_module.n_jobs
    experiment_name = config_module.experiment_name
    groups_csv = config_module.groups_csv
    training_site = config_module.training_site


    random.seed(N_SEED)
    np.random.seed(N_SEED)
    parameters = {'C': C_values}

    print("Loading data....")
    file_npz = np.load("./kernels/" + precomputed_kernel_files)
    y = file_npz['labels']
    precomputed_kernel = file_npz['kernel']

    print("Loading sites information...")
    groups = np.loadtxt(groups_csv, delimiter=',')
    print("Found ", np.unique(groups).size, " different sites.")
    print("Training on ", training_site, " site.")


    if not os.path.exists("./results/"+experiment_name+"/single/"):
        os.makedirs("./results/"+experiment_name+"/single/")

    permutation_classifiers = []

    grid_scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)

    train_index = np.where(groups==training_site)[0]

    x_train = precomputed_kernel[train_index, :][:, train_index]
    y_train = y[train_index]

    y_train = np.ravel(y_train)

    svc = SVC(kernel='precomputed', random_state=1)
    nested_skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=1)
    grid = GridSearchCV(svc, parameters, cv=nested_skf, scoring = grid_scorer, n_jobs=n_jobs, refit=True, verbose=3)
    grid.fit(x_train, y_train)
    best_svc = grid.best_estimator_

    y_predicted = best_svc.predict(precomputed_kernel[:, :][:, train_index])

    nb_sites = 2
    site_test_bac = np.zeros((nb_sites,))
    for i in range(nb_sites):
        print("")
        print("Confusion matrix of site ", i)
        cm = confusion_matrix(y[groups==i], y_predicted[groups==i])
        print(cm)

        test_bac = np.sum(np.true_divide(np.diagonal(cm), np.sum(cm, axis=1))) / cm.shape[1]
        test_sens = np.true_divide(cm[1, 1], np.sum(cm[1, :]))
        test_spec = np.true_divide(cm[0, 0], np.sum(cm[0, :]))
        error_rate = np.true_divide(cm[0, 1] + cm[1, 0], np.sum(np.sum(cm)))

        print("Balanced acc: %.4f " % (test_bac))
        print("Sensitivity: %.4f " % (test_sens))
        print("Specificity: %.4f " % (test_spec))
        print("Error Rate: %.4f " % (error_rate))

        site_test_bac[i] = test_bac


    permutation_classifiers.append(best_svc)

    f = open("./results/"+experiment_name+"/single/best_clf.pkl", 'wb')
    pickle.dump(permutation_classifiers, f)
    f.close()
    np.savez("./results/"+experiment_name+"/single/final_BAC.npz", final_BAC=site_test_bac)
    np.savez("./results/"+experiment_name+"/single/predictions.npz", y_predicted=y_predicted, y = y)

    return site_test_bac


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
        best_seed = 0
        best_perf = 0
        for seed in range(config_module.N_SEED[0],config_module.N_SEED[1]):
            perf = main(config_module, seed)
            if perf > best_perf:
                best_seed = seed
                best_perf = perf
        print("BEST SEED IS ", best_seed)
        print("WITH MEAN PERFORMANCE OF", best_perf)