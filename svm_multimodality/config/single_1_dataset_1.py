# -------------------------------------------------------------------------
# Modalities
# -------------------------------------------------------------------------
precomputed_kernel_files = ["functional_matrix_kernel.npz"]
# -------------------------------------------------------------------------

n_jobs = 9

# -------------------------------------------------------------------------
experiment_name = "1_single_COBRE" # do not use spaces
N_SEED = 1
# N_SEED = [1, 100] #To find the best seed between 1 and 100
C_values = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
cv_n_folds = 10
nested_cv_n_folds = 5

# ---------------------ONLY USED ON PERMUTATION TEST----------------------------
n_permutations = 100

# ---------------------ONLY USED ON discriminant features ----------------------------
data_dirs = ["./data/KCL3/"]
type_data = ["MATRIX"] #use "MATRIX" or "3D"


