# -------------------------------------------------------------------------
# Modalities
# -------------------------------------------------------------------------
precomputed_kernel_files = ["test.npz"]         # Select the pooled kernel

# -------------------------------------------------------------------------

n_jobs = 9

# -------------------------------------------------------------------------
experiment_name = "1_single_onevsall_train1" # do not use spaces
N_SEED = 1
# N_SEED = [1, 100] #To find the best seed between 1 and 100
C_values = [1e-5, 1e-4, ] # INCREASED SPACE OF PARAMETER TO TRY TO IMPROVE PERFORMANCE 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5
cv_n_folds = 10                                                     #USE ONLY THIS ONE
nested_cv_n_folds = 10

# ---------------------ONLY USED ON PERMUTATION TEST----------------------------
n_permutations = 100

# ---------------------ONLY USED ON discriminant features ----------------------------
data_dirs = ["./data/KCL3/"]
type_data = ["MATRIX"] #use "MATRIX" or "3D"

# ---------------------ONLY USED ON onevsall -------------------------
groups_csv = "./data/sites.csv"
training_site = 1

