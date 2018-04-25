from GaussianMixtureModel import GaussianMixtureModel
from KMeansModel import KMeansModel
import numpy as np
import matplotlib.pyplot as plt

## INIT ##
np.random.seed(18)
Covariances = ["full", "tied", "spherical", "diag"]
Scaling_Types = ["standard", "robust", "normalizer", "minmax"]

## DATA IMPORT ##
data = np.genfromtxt('seeds_dataset.txt', usecols=range(7))
labels = np.genfromtxt('seeds_dataset.txt', usecols=7)

## MODEL INSTANTIATION ##
GMM = GaussianMixtureModel(data, labels)
KM = KMeansModel(data, labels)

## RUN ##

# Gaussian mixture
index = 1
for covariance_type in Covariances:
    for scaling_type in Scaling_Types:
        np.random.seed(18)
        GMM.run(subplot_ = plt.subplot(len(Covariances), len(Scaling_Types), index), covariance_type=covariance_type, scaling_type=scaling_type)
        GMM.reset()
        index+=1
plt.show()

# KMeans
index = 1
for scaling_type in Scaling_Types:
    np.random.seed(18)
    KM.run(subplot_ = plt.subplot(1, len(Scaling_Types), index), scaling_type=scaling_type)
    KM.reset()
    index+=1
plt.show()
