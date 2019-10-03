from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar

# ---------------------------------------------------------------------------------
# CHANGE HERE
# ---------------------------------------------------------------------------------

predictions1 = "./results/2_single_COBRE/single/predictions.npz"
predictions2 = "./results/2_single_COBRE/single/predictions.npz"

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

pred1_file = np.load(predictions1)
prediction_1 = pred1_file["y_predictions"]

pred1 = prediction_1[0]
for i in range(1, len(prediction_1)):
    pred1 = np.hstack((pred1, prediction_1[i]))


y_true_list = pred1_file["y_true"]

y_true = y_true_list[0]
for i in range(1, len(y_true_list)):
    y_true = np.hstack((y_true, y_true_list[i]))


pred2_file = np.load(predictions2)
prediction_2 = pred2_file["y_predictions"]

pred2 = prediction_2[0]
for i in range(1, len(prediction_2)):
    pred2 = np.hstack((pred2, prediction_2[i]))


tb = mcnemar_table(y_target=y_true,
                   y_model1=pred1,
                   y_model2=pred2)


chi2, p = mcnemar(ary=tb, corrected=True)
print('chi-squared:', chi2)
print('p-value:', p)
