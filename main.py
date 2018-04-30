from GaussianMixtureModel import GaussianMixtureModel
from KMeansModel import KMeansModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

## INIT ##
np.random.seed(18)
Covariances = ["full", "tied", "spherical", "diag"]
Scaling_Types = ["standard", "robust", "normalizer", "minmax"]
KM_top_orders = [False, False, True, True]
GM_seeds = [18, 8, 12, 8]
KM_seeds = [2,0,16,6]
GM_top_orders = [False, False, False, False]
GM_top_covar = ["diag", "diag", "tied", "diag"]

## DATA IMPORT ##
data = np.genfromtxt('seeds_dataset.txt', usecols=range(7))
labels = np.genfromtxt('seeds_dataset.txt', usecols=7)

## MODEL INSTANTIATION ##
GMM = GaussianMixtureModel(data, labels)
KM = KMeansModel(data, labels)

## RUN ##

# Gaussian mixture
index = 0
for scaling_type in Scaling_Types:
    np.random.seed(GM_seeds[index])
    GMM.run(subplot_ = plt.subplot(2, 2, index+1), covariance_type = GM_top_covar[index], scaling_type=scaling_type, reduce_first = GM_top_orders[index])
    plt.title("Scaling: {}  Covar: {}  Reduce first: {}\nScore:{}".format(scaling_type, GM_top_covar[index], GM_top_orders[index], GMM.score))
    GMM.reset()
    index+=1
plt.suptitle("Top scoring gaussian mixture models for 2 components")
plt.show()

# KMeans
index = 0
for scaling_type in Scaling_Types:
    np.random.seed(KM_seeds[index])
    KM.run(subplot_ = plt.subplot(2, 2, index+1), scaling_type=scaling_type, reduce_first = KM_top_orders[index])
    plt.title("Scaling: {}     Reduce first: {} Score: {}%".format(scaling_type,KM_top_orders[index],KM.score))
    KM.reset()
    index+=1
plt.suptitle("Top scoring k-means models for 2 components")
plt.show()
