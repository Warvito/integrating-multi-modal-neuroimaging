from __future__ import print_function
import argparse
import imp
import random
import numpy as np

from shogun import BinaryLabels
from shogun import CombinedKernel,  CustomKernel
from shogun import MKLClassification

import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


def main(config_module, N_SEED):
    C_values = config_module.C_values
    norm_values = config_module.norm_values
    nested_cv_n_folds = config_module.nested_cv_n_folds
    cv_n_folds = config_module.cv_n_folds
    precomputed_kernel_files = config_module.precomputed_kernel_files

    random.seed(N_SEED)
    np.random.seed(N_SEED)

    file_npz = np.load("./kernels/" + precomputed_kernel_files[0])
    y = file_npz['labels']
    y = (y * 2) - 1
    y = y.astype('float64')

    skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=N_SEED)

    cv_test_bac = np.zeros((cv_n_folds,))
    cv_test_sens = np.zeros((cv_n_folds,))
    cv_test_spec = np.zeros((cv_n_folds,))
    cv_error_rate = np.zeros((cv_n_folds,))

    kernels = []
    print("Loading kernels...")
    for precomputed_kernel_file in precomputed_kernel_files:
        file_npz = np.load("./kernels/" + precomputed_kernel_file)
        kernels.append(file_npz['kernel'])

    print("Starting Stratified Cross Validation with ", cv_n_folds, " folds")
    for i, (train_index, test_index) in enumerate(skf.split(y, y)):
        start_time = time.time()

        print("")
        print("Fold ", i)
        y_train, y_test = y[train_index], y[test_index]

        best_C = 1
        best_norm = 1

        best_performance = 0
        for C in C_values:
            for norm in norm_values:
                fold_prediction = np.zeros((nested_cv_n_folds,))

                nested_skf = StratifiedKFold(n_splits=nested_cv_n_folds, shuffle=True, random_state=1)
                for j, (train_index2, val_index) in enumerate(nested_skf.split(y_train, y_train)):
                    y_train2, y_val = y_train[train_index2], y_train[val_index]

                    # set up
                    kernel = CombinedKernel()
                    labels = BinaryLabels(y_train2)

                    for k in range(len(kernels)):
                        precomputed_kernel = kernels[k]

                        x_train2, x_val = precomputed_kernel[train_index[train_index2], :][:,
                                          train_index[train_index2]], precomputed_kernel[train_index[val_index],
                                                                      :][:, train_index[train_index2]]


                        ##################################
                        # Kernel Custom
                        subkernel = CustomKernel(x_train2)
                        kernel.append_kernel(subkernel)

                    elasticnet_lambda = 0
                    # which norm to use for MKL
                    mkl_norm = norm
                    mkl_epsilon = 1e-5
                    # Cost C MKL
                    C_mkl = 0
                    # Creating model
                    mkl = MKLClassification()
                    mkl.set_elasticnet_lambda(elasticnet_lambda)
                    mkl.set_C_mkl(C_mkl)
                    mkl.set_mkl_norm(mkl_norm)
                    mkl.set_mkl_epsilon(mkl_epsilon)
                    # set cost (neg, pos)
                    mkl.set_C(C, C)
                    # set kernel and labels
                    mkl.set_kernel(kernel)
                    mkl.set_labels(labels)

                    ##################################
                    # Train
                    mkl.train()

                    ##################################
                    # Test
                    kernel = CombinedKernel()
                    for k in range(len(kernels)):
                        precomputed_kernel = kernels[k]

                        x_train2, x_val = precomputed_kernel[train_index[train_index2], :][:,
                                          train_index[train_index2]], precomputed_kernel[train_index[val_index],
                                                                      :][:, train_index[train_index2]]


                        ##################################
                        # Kernel Custom
                        subkernel = CustomKernel(x_val.T)
                        kernel.append_kernel(subkernel)


                    # Predicts
                    mkl.set_kernel(kernel)
                    prediction = mkl.apply().get_labels()

                    cm = confusion_matrix(y_val, prediction)
                    test_bac = np.sum(np.true_divide(np.diagonal(cm), np.sum(cm, axis=1))) / cm.shape[1]
                    fold_prediction[j] = test_bac

                if np.mean(fold_prediction) > best_performance:
                    best_performance = np.mean(fold_prediction)
                    best_C = C
                    best_norm = norm


        # set up
        kernel = CombinedKernel()
        labels = BinaryLabels(y_train)
        for j in range(len(kernels)):
            precomputed_kernel = kernels[j]

            x_train, x_test = precomputed_kernel[train_index, :][:, train_index], precomputed_kernel[test_index, :][:, train_index]

            ##################################
            # Kernel Custom
            subkernel = CustomKernel(x_train)
            kernel.append_kernel(subkernel)

        elasticnet_lambda = 0
        # which norm to use for MKL
        mkl_norm = best_norm
        mkl_epsilon = 1e-5
        # Cost C MKL
        C_mkl = 0
        # Creating model
        mkl = MKLClassification()
        mkl.set_elasticnet_lambda(elasticnet_lambda)
        mkl.set_C_mkl(C_mkl)
        mkl.set_mkl_norm(mkl_norm)
        mkl.set_mkl_epsilon(mkl_epsilon)
        # set cost (neg, pos)
        mkl.set_C(best_C, best_C)
        # set kernel and labels
        mkl.set_kernel(kernel)
        mkl.set_labels(labels)

        ##################################
        # Train
        mkl.train()

        ##################################
        # Test
        kernel = CombinedKernel()
        for k in range(len(kernels)):
            precomputed_kernel = kernels[k]

            x_train, x_test = precomputed_kernel[train_index, :][:, train_index], precomputed_kernel[test_index, :][:, train_index]


            ##################################
            # Kernel Custom
            subkernel = CustomKernel(x_test.T)
            kernel.append_kernel(subkernel)

        # Predicts
        mkl.set_kernel(kernel)
        prediction = mkl.apply().get_labels()


        print("")
        print("Confusion matrix")
        cm = confusion_matrix(y_test,prediction)
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
        stop_time = time.time()
        print("--- %s seconds ---" % (stop_time - start_time))
        print("ETA: ", (i-cv_n_folds)*(stop_time - start_time), " seconds")

    print("")
    print("")
    print("Cross-validation balanced acc: %.4f +- %.4f" % (cv_test_bac.mean(), cv_test_bac.std()))
    print("Cross-validation Sensitivity: %.4f +- %.4f" % (cv_test_sens.mean(), cv_test_sens.std()))
    print("Cross-validation Specificity: %.4f +- %.4f" % (cv_test_spec.mean(), cv_test_spec.std()))
    print("Cross-validation Error Rate: %.4f +- %.4f" % (cv_error_rate.mean(), cv_error_rate.std()))
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
