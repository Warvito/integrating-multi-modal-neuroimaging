# -------------------------------------------------------------------------
# Modalities
# -------------------------------------------------------------------------
precomputed_kernel_files = ["test.npz", "test1.npz"]
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
experiment_name = "1_dataset_COBRE" # do not use spaces
N_SEED = 1
# N_SEED = [1, 100] #To find the best seed between 1 and 100
C_values = [1e-3, 1e-2]
cv_n_folds = 2
nested_cv_n_folds = 10

# ---------------------ONLY USED WITH AV_SVM ----------------------------------
AV_SVM_weigths = [1,10]
n_jobs = 9

# ---------------------ONLY USED WITH MKL ----------------------------------
norm_values = [1, 2, 3, 4, 5]

# ---------------------ONLY USED ON PERMUTATION TEST----------------------------
n_permutations = 100

# ---------------------ONLY USED ON discriminant features ----------------------------
data_dirs = ["./data/KCL3/",  "./data/SC3/"]
type_data = ["MATRIX", "MATRIX"] #use "MATRIX" or "3D"


# ---------------------ONLY USED ON onevsall -------------------------
groups_csv = "./data/sites.csv"
training_site = 1
