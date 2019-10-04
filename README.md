# Integrating machining learning and multi-modal neuroimaging to detect schizophrenia at the level of the individual
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/Warvito/integrating-multi-modal-neuroimaging/blob/master/LICENSE)

Official script of the paper Integrating machining learning and multi-modal neuroimaging to detect schizophrenia at the level of the individual implemented by Du Lei and Walter Hugo Lopez Pinaya

## Abstract
Schizophrenia is a severe psychiatric disorder associated with both structural and
functional brain abnormalities. In the past few years, there has been growing interest
in the application of machine learning techniques to neuroimaging data for the
diagnostic and prognostic assessment of this disorder. However, the vast majority of
studies published so far have used either structural or functional neuroimaging data,
without accounting for the multi-modal nature of the disorder. Structural MRI and
resting state functional MRI data were acquired from a total of 295 patients with
schizophrenia and 452 healthy controls at five research centers. We extracted features
from the data including gray matter volume, white matter volume, amplitude of low-
frequency fluctuation, regional homogeneity and two connectome-wide based
metrics: structural covariance matrices and functional connectivity matrices. A
support vector machine classifier was trained on each dataset separately to distinguish
the subjects at individual level using each of the single feature as well as their
combination, and 10-fold cross-validation approach was used to investigate the
performance of the model. Functional data allow higher accuracy of classification
than structural data (mean 82.75% vs. 75.84%). Within each modality, the
combination of images and matrices improves performance, resulting in mean
accuracies of 81.63% for structural data and 87.59% for functional data. The use of all
combined structural and functional measures allows the highest accuracy of
classification (90.83%). We conclude that combining multi-modal measures within a
single model is a promising direction for developing biologically-informed diagnostic
tools in schizophrenia.



## Requirements
- Python 2
- [Numpy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scikit-learn](https://scikit-learn.org/)


## Installing the dependencies
Install virtualenv and creating a new virtual environment:

    pip install virtualenv
    virtualenv -p /usr/bin/python2 ./venv

Install dependencies

    pip install -r requirements.txt


## Citation
If you find this code useful for your research, please cite:

    @article{}
 