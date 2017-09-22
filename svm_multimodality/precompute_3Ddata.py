import glob
import numpy as np
import nibabel as nib

# ---------------------------------------------------------------------------------
# CHANGE HERE
# ---------------------------------------------------------------------------------

data_dir = './data/IMG/'
labels_path = "./data/Labels.csv"
kernel_file = './kernels/img.npz'
input_data_type = ".nii"



# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

print "Reading labels from %s" % labels_path
labels = np.genfromtxt(labels_path, delimiter=',', dtype='int8')
print "   # of labels samples: %d " % len(labels)
print "Reading images with format {} from: %s".format(input_data_type, data_dir)
paths_train = glob.glob(data_dir + "/*" + input_data_type)
paths_train.sort()

n_samples = len(labels)
if n_samples != len(paths_train):
    raise ValueError('Different number of labels and images files')

print "Loading images"
print "   # of images samples: %d " % len(paths_train)

n_samples = len(paths_train)

print n_samples

K = np.float64(np.zeros((n_samples, n_samples)))
step_size = 30
images = []

# outer loop
for i in range(int(np.ceil(n_samples / np.float(step_size)))):

    it = i + 1
    max_it = int(np.ceil(n_samples / np.float(step_size)))
    print " outer loop iteration: %d of %d." % (it, max_it)

    # generate indices and then paths for this block
    start_ind_1 = i * step_size
    stop_ind_1 = min(start_ind_1 + step_size, n_samples)
    block_paths_1 = paths_train[start_ind_1:stop_ind_1]

    # read in the images in this block
    images_1 = []
    for k, path in enumerate(block_paths_1):
        img = nib.load(path)
        img = img.get_data()
        img = np.asarray(img, dtype='float64')
        img_vec = np.reshape(img, np.product(img.shape))
        images_1.append(img_vec)
        del img
    images_1 = np.array(images_1)
    for j in range(i + 1):

        it = j + 1
        max_it = i + 1

        print " inner loop iteration: %d of %d." % (it, max_it)

        # if i = j, then sets of image data are the same - no need to load
        if i == j:

            start_ind_2 = start_ind_1
            stop_ind_2 = stop_ind_1
            images_2 = images_1

        # if i !=j, read in a different block of images
        else:
            start_ind_2 = j * step_size
            stop_ind_2 = min(start_ind_2 + step_size, n_samples)
            block_paths_2 = paths_train[start_ind_2:stop_ind_2]

            images_2 = []
            for k, path in enumerate(block_paths_2):
                img = nib.load(path)
                img = img.get_data()
                img = np.asarray(img, dtype='float64')
                img_vec = np.reshape(img, np.product(img.shape))
                images_2.append(img_vec)
                del img
            images_2 = np.array(images_2)

        block_K = np.dot(images_1, np.transpose(images_2))
        K[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = block_K
        K[start_ind_2:stop_ind_2, start_ind_1:stop_ind_1] = np.transpose(block_K)

print ""
print "Saving Dataset"
print "   Kernel+Labels:" + kernel_file
np.savez(kernel_file, kernel=K, labels=labels)
print "Done"
