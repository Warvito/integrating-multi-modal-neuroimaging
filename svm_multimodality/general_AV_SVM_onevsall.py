from __future__ import print_function
import argparse
import imp
import random
import numpy as np
import itertools
import os
import pickle

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix


def balanced_accuracy_score(actual, prediction):
    cm = confusion_matrix(actual, prediction)
    bac = np.sum(np.true_divide(np.diagonal(cm), np.sum(cm, axis=1))) / cm.shape[1]
    return bac


# TODO: Progress bar
# TODO: ETA
def main(config_module, N_SEED):
    C_values = config_module.C_values
    nested_cv_n_folds = config_module.nested_cv_n_folds
    cv_n_folds = config_module.cv_n_folds
    precomputed_kernel_files = config_module.precomputed_kernel_files
    n_jobs = config_module.n_jobs
    AV_SVM_weigths = config_module.AV_SVM_weigths
    experiment_name = config_module.experiment_name

    groups_csv = config_module.groups_csv
    training_site = config_module.training_site

    random.seed(N_SEED)
    np.random.seed(N_SEED)
    parameters = {'C': C_values}

    file_npz = np.load("./kernels/" + precomputed_kernel_files[0])
    y = file_npz['labels']
    y= np.ravel(y)

    print("Loading sites information...")
    groups = np.loadtxt(groups_csv, delimiter=',')
    print("Found ", np.unique(groups).size, " different sites.")
    print("Training on ", training_site, " site.")


    if not os.path.exists("./results/"+experiment_name+"/AV/"):
        os.makedirs("./results/"+experiment_name+"/AV/")

    train_index = np.where(groups==training_site)[0]

    grid_scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)

    permutation_classifiers = []
    permutation_AV_combination = []

    kernels = []
    print("Loading kernels...")
    for precomputed_kernel_file in precomputed_kernel_files:
        file_npz = np.load("./kernels/" + precomputed_kernel_file)
        kernels.append(file_npz['kernel'])


    y_train = y[train_index]

    best_classifiers = []
    for j in range(len(kernels)):
        precomputed_kernel = kernels[j]

        x_train, x_test = precomputed_kernel[train_index, :][:, train_index], precomputed_kernel[:, :][:, train_index]

        svc = SVC(kernel='precomputed', probability=True, random_state=1)
        nested_skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=1)
        grid = GridSearchCV(svc, parameters, cv=nested_skf, scoring=grid_scorer, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        best_classifiers.append(grid.best_estimator_)

    AV_weights = []
    for _ in range(len(precomputed_kernel_files)):
        AV_weights.append(range(AV_SVM_weigths[0], AV_SVM_weigths[1] + 1))


    best_performance = 0
    best_combination = 0
    for combination in itertools.product(*AV_weights):
        fold_prediction = np.zeros((cv_n_folds,))

        nested_skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=1)
        for j, (train_index2, val_index) in enumerate(nested_skf.split(y_train, y_train)):
            predictions_proba = []
            for k in range(len(kernels)):
                precomputed_kernel = kernels[k]

                x_train2, x_val = precomputed_kernel[train_index[train_index2], :][:, train_index[train_index2]], precomputed_kernel[train_index[val_index],
                                                                                      :][:, train_index[train_index2]]
                y_train2, y_val = y_train[train_index2], y_train[val_index]

                (best_classifiers[k]).fit(x_train2,y_train2)

                predictions_proba.append((best_classifiers[k]).predict_proba(x_val))

            proba_sum = 0
            for k in range(len(combination)):
                proba_sum += combination[k]*np.array(predictions_proba[k])

            cm = confusion_matrix(y_val, np.argmax(proba_sum, axis=-1))
            test_bac = np.sum(np.true_divide(np.diagonal(cm), np.sum(cm, axis=1))) / cm.shape[1]
            fold_prediction[j] = test_bac

        if np.mean(fold_prediction) > best_performance:
            best_performance = np.mean(fold_prediction)
            best_combination = combination

    predictions_proba = []
    for j in range(len(kernels)):
        precomputed_kernel = kernels[j]

        x_train, x_test = precomputed_kernel[train_index, :][:, train_index], precomputed_kernel[:, :][:, train_index]
        (best_classifiers[j]).fit(x_train, y_train)

        predictions_proba.append((best_classifiers[j]).predict_proba(x_test))


    proba_sum = 0
    for k in range(len(best_combination)):
        proba_sum += best_combination[k] * np.array(predictions_proba[k])


    y_predicted =np.argmax(proba_sum, axis=-1)
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


    permutation_classifiers.append(best_classifiers)
    permutation_AV_combination.append(best_combination)

    f = open("./results/"+experiment_name+"/AV//best_clf.pkl", 'wb')
    pickle.dump(permutation_classifiers, f)
    f.close()
    f = open("./results/"+experiment_name+"/AV/best_AV_combination.pkl", 'wb')
    pickle.dump(permutation_AV_combination, f)
    f.close()
    np.savez("./results/"+experiment_name+"/AV/final_BAC.npz", final_BAC=site_test_bac)
    np.savez("./results/"+experiment_name+"/AV/predictions.npz", y_predicted=y_predicted, y = y)

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
