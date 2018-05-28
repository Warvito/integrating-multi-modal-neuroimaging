from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
import matplotlib.cm as cm

# ---------------------------------------------------------------------------------
# CHANGE HERE
# ---------------------------------------------------------------------------------

data_dir = "./data/SC3"
sites_path = "./data/sites2.csv"
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

matrix_files = glob.glob(data_dir + "/*" + input_data_type)

matrix_files = sort_nicely(matrix_files)
n_samples = len(matrix_files)

data = []
for file_path in matrix_files:
    matrix = np.genfromtxt(file_path)
    matrix = lower_tri_witout_diag(matrix)
    matrix = np.asarray(matrix, dtype='float32')
    data.append(matrix)

data = np.array(data, dtype="float32")

sites_df = pd.read_csv(sites_path)
sites = sites_df.values
n_components = 2
perplexities = [2, 5, 20, 30, 50, 70, 100, 150, 200]
random_state = range(10)
colors = cm.rainbow(np.linspace(0, 1, len(np.unique(sites))))


for i in range(4):
    (fig, subplots) = plt.subplots(3, 3)
    fig.suptitle('Random State ' + str(i))
    perp = 2
    ax = subplots[0][0]
    ax.set_title('Perplexity ' + str(perp))
    tsne = manifold.TSNE(n_components=n_components,
                         init='random',
                         learning_rate=200.0,
                         n_iter=2000,
                         random_state=i,
                         perplexity=perp)

    X_embedded = tsne.fit_transform(data)
    for j, site in enumerate(np.unique(sites)):
        mask = np.squeeze(sites==site, axis=1)
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], color = colors[j,:])

    perp = 3
    ax = subplots[0][1]
    ax.set_title('Perplexity ' + str(perp))
    tsne = manifold.TSNE(n_components=n_components,
                         init='random',
                         learning_rate=200.0,
                         n_iter=2000,
                         random_state=i,
                         perplexity=perp)

    X_embedded = tsne.fit_transform(data)
    for j, site in enumerate(np.unique(sites)):
        mask = np.squeeze(sites==site, axis=1)
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], color = colors[j,:])

    perp = 4
    ax = subplots[0][2]
    ax.set_title('Perplexity ' + str(perp))
    tsne = manifold.TSNE(n_components=n_components,
                         init='random',
                         learning_rate=200.0,
                         n_iter=2000,
                         random_state=i,
                         perplexity=perp)

    X_embedded = tsne.fit_transform(data)
    for j, site in enumerate(np.unique(sites)):
        mask = np.squeeze(sites==site, axis=1)
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], color = colors[j,:])

    perp = 6
    ax = subplots[1][0]
    ax.set_title('Perplexity ' + str(perp))
    tsne = manifold.TSNE(n_components=n_components,
                         init='random',
                         learning_rate=200.0,
                         n_iter=2000,
                         random_state=i,
                         perplexity=perp)

    X_embedded = tsne.fit_transform(data)
    for j, site in enumerate(np.unique(sites)):
        mask = np.squeeze(sites==site, axis=1)
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], color = colors[j,:])

    perp = 7
    ax = subplots[1][1]
    ax.set_title('Perplexity ' + str(perp))
    tsne = manifold.TSNE(n_components=n_components,
                         init='random',
                         learning_rate=200.0,
                         n_iter=2000,
                         random_state=i,
                         perplexity=perp)

    X_embedded = tsne.fit_transform(data)
    for j, site in enumerate(np.unique(sites)):
        mask = np.squeeze(sites==site, axis=1)
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], color = colors[j,:])

    perp = 8
    ax = subplots[1][2]
    ax.set_title('Perplexity ' + str(perp))
    tsne = manifold.TSNE(n_components=n_components,
                         init='random',
                         learning_rate=200.0,
                         n_iter=2000,
                         random_state=i,
                         perplexity=perp)

    X_embedded = tsne.fit_transform(data)
    for j, site in enumerate(np.unique(sites)):
        mask = np.squeeze(sites==site, axis=1)
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], color = colors[j,:])

    perp = 9
    ax = subplots[2][0]
    ax.set_title('Perplexity ' + str(perp))
    tsne = manifold.TSNE(n_components=n_components,
                         init='random',
                         learning_rate=200.0,
                         n_iter=2000,
                         random_state=i,
                         perplexity=perp)

    X_embedded = tsne.fit_transform(data)
    for j, site in enumerate(np.unique(sites)):
        mask = np.squeeze(sites==site, axis=1)
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], color = colors[j,:])

    perp = 10
    ax = subplots[2][1]
    ax.set_title('Perplexity ' + str(perp))
    tsne = manifold.TSNE(n_components=n_components,
                         init='random',
                         learning_rate=200.0,
                         n_iter=2000,
                         random_state=i,
                         perplexity=perp)

    X_embedded = tsne.fit_transform(data)
    for j, site in enumerate(np.unique(sites)):
        mask = np.squeeze(sites==site, axis=1)
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], color = colors[j,:])

    perp = 11
    ax = subplots[2][2]
    ax.set_title('Perplexity ' + str(perp))
    tsne = manifold.TSNE(n_components=n_components,
                         init='random',
                         learning_rate=200.0,
                         n_iter=2000,
                         random_state=i,
                         perplexity=perp)

    X_embedded = tsne.fit_transform(data)
    for j, site in enumerate(np.unique(sites)):
        mask = np.squeeze(sites==site, axis=1)
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], color = colors[j,:])

    plt.show()

