import glob
import re
import numpy as np


# ---------------------------------------------------------------------------------
# CHANGE HERE
# ---------------------------------------------------------------------------------

data_dir = "./data/SC3"
labels_path = "./data/Labels.csv"
sites_path = "./data/sites3.csv"
kernel_file = './kernels/functional_matrix_kernel.npz'
input_data_type = ".txt"


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    return sorted(l, key=alphanum_key)

def lower_tri_witout_diag(x):
    return np.concatenate([ x[i][:i] for i in range(x.shape[0])])

labels = np.genfromtxt(labels_path, dtype="int32")
sites = np.genfromtxt(sites_path, dtype="int32")
matrix_files = glob.glob(data_dir + "/*" + input_data_type)

matrix_files = sort_nicely(matrix_files)
n_samples = len(matrix_files)

K = np.float64(np.zeros((n_samples, n_samples)))
step_size = 30

# outer loop
for i in range(int(np.ceil(n_samples / np.float(step_size)))):
    it = i + 1
    max_it = int(np.ceil(n_samples / np.float(step_size)))
    print(" outer loop iteration: %d of %d." % (it, max_it))

    # generate indices and then paths for this block
    start_ind_1 = i * step_size
    stop_ind_1 = min(start_ind_1 + step_size, n_samples)
    block_paths_1 = matrix_files[start_ind_1:stop_ind_1]

    # read in the images in this block
    images_1 = []
    for k, path in enumerate(block_paths_1):
        matrix = np.genfromtxt(path)
        matrix = lower_tri_witout_diag(matrix)
        matrix = np.asarray(matrix, dtype='float64')
        img_vec = np.reshape(matrix, np.product(matrix.shape))
        images_1.append(img_vec)
    images_1 = np.array(images_1)
    for j in range(i + 1):

        it = j + 1
        max_it = i + 1

        print(" inner loop iteration: %d of %d." % (it, max_it))

        # if i = j, then sets of image data are the same - no need to load
        if i == j:

            start_ind_2 = start_ind_1
            stop_ind_2 = stop_ind_1
            images_2 = images_1

        # if i !=j, read in a different block of images
        else:
            start_ind_2 = j * step_size
            stop_ind_2 = min(start_ind_2 + step_size, n_samples)
            block_paths_2 = matrix_files[start_ind_2:stop_ind_2]

            images_2 = []
            for k, path in enumerate(block_paths_2):
                matrix = np.genfromtxt(path)
                matrix = lower_tri_witout_diag(matrix)
                matrix = np.asarray(matrix, dtype='float64')
                img_vec = np.reshape(matrix, np.product(matrix.shape))
                images_2.append(img_vec)

            images_2 = np.array(images_2)

        block_K = np.dot(images_1, np.transpose(images_2))
        K[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = block_K
        K[start_ind_2:stop_ind_2, start_ind_1:stop_ind_1] = np.transpose(block_K)

print("")
print("Saving Dataset")
print("   Kernel+Labels:" + kernel_file)
np.savez(kernel_file, kernel=K, labels=labels, sites=sites)
print("Done")