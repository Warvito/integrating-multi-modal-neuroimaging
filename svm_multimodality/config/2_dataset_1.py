# -------------------------------------------------------------------------
# Modalities
# -------------------------------------------------------------------------
precomputed_kernel_files = ["functional_matrix_kernel.npz", "img.npz"]
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
experiment_name = "2_dataset_COBRE" # do not use spaces
N_SEED = 1
# N_SEED = [1, 100] #To find the best seed between 1 and 100
C_values = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
cv_n_folds = 10
nested_cv_n_folds = 5

# ---------------------ONLY USED WITH AV_SVM ----------------------------------
AV_SVM_weigths = [1,10]
n_jobs = 9

# ---------------------ONLY USED WITH MKL ----------------------------------
norm_values = [1, 2, 3, 4, 5]

# ---------------------ONLY USED ON PERMUTATION TEST----------------------------
n_permutations = 100

# ---------------------ONLY USED ON discriminant features ----------------------------
data_dirs = ["./data/KCL3/",  "./data/IMG/"]
type_data = ["MATRIX", "3D"] #use "MATRIX" or "3D"


