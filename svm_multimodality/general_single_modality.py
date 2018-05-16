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
    nested_cv_n_folds = config_module.nested_cv_n_folds
    cv_n_folds = config_module.cv_n_folds
    precomputed_kernel_files = config_module.precomputed_kernel_files[0]
    n_jobs = config_module.n_jobs
    experiment_name = config_module.experiment_name

    random.seed(N_SEED)
    np.random.seed(N_SEED)
    parameters = {'C': C_values}

    print("Loading data....")
    file_npz = np.load("./kernels/" + precomputed_kernel_files)
    y = file_npz['labels']
    precomputed_kernel = file_npz['kernel']


    if not os.path.exists("./results/"+experiment_name+"/single/"):
        os.makedirs("./results/"+experiment_name+"/single/")

    permutation_classifiers = []

    grid_scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)

    skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=N_SEED)

    cv_test_bac = np.zeros((cv_n_folds,))
    cv_test_sens = np.zeros((cv_n_folds,))
    cv_test_spec = np.zeros((cv_n_folds,))
    cv_error_rate = np.zeros((cv_n_folds,))
    i=0
    for train_index, test_index in skf.split(y, y):
        x_train, x_test = precomputed_kernel[train_index, :][:, train_index], precomputed_kernel[test_index, :][:, train_index]
        y_train, y_test = y[train_index], y[test_index]


        svc = SVC(kernel='precomputed', random_state=1)
        nested_skf = StratifiedKFold(n_splits=nested_cv_n_folds, shuffle=True, random_state=1)
        grid = GridSearchCV(svc, parameters, cv=nested_skf, scoring = grid_scorer, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        best_svc = SVC(kernel='precomputed', C= grid.best_estimator_.C, random_state=1)
        best_svc.fit(x_train, y_train)

        y_predicted = best_svc.predict(x_test)


        print("")
        print("Confusion matrix")
        cm = confusion_matrix(y_test, y_predicted)
        print(cm)

        test_bac = np.sum(np.true_divide(np.diagonal(cm), np.sum(cm, axis=1))) / cm.shape[1]
        test_sens = np.true_divide(cm[1, 1], np.sum(cm[1, :]))
        test_spec = np.true_divide(cm[0, 0], np.sum(cm[0, :]))
        error_rate = np.true_divide(cm[0, 1] + cm[1, 0], np.sum(np.sum(cm)))

        print("Balanced acc: %.4f " % (test_bac))
        print("Sensitivity: %.4f " % (test_sens))
        print("Specificity: %.4f " % (test_spec))
        print("Error Rate: %.4f " % (error_rate))

        cv_test_bac[i] = test_bac
        cv_test_sens[i] = test_sens
        cv_test_spec[i] = test_spec
        cv_error_rate[i] = error_rate

        permutation_classifiers.append(best_svc)

        i += 1
    print("")
    print("")
    print("Cross-validation balanced acc: %.4f +- %.4f" % (cv_test_bac.mean(), cv_test_bac.std()))
    print("Cross-validation Sensitivity: %.4f +- %.4f" % (cv_test_sens.mean(), cv_test_sens.std()))
    print("Cross-validation Specificity: %.4f +- %.4f" % (cv_test_spec.mean(), cv_test_spec.std()))
    print("Cross-validation Error Rate: %.4f +- %.4f" % (cv_error_rate.mean(), cv_error_rate.std()))

    f = open("./results/"+experiment_name+"/single/best_clf.pkl", 'wb')
    pickle.dump(permutation_classifiers, f)
    f.close()
    f = open("./results/"+experiment_name+"/single/final_BAC.pkl", 'wb')
    pickle.dump(cv_test_bac.mean(), f)
    f.close()

    return(cv_test_bac.mean())


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