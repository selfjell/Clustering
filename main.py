from GaussianMixtureModel import GaussianMixtureModel
from KMeansModel import KMeansModel
import numpy as np
import matplotlib.pyplot as plt

## INIT ##
np.random.seed(18)

## DATA IMPORT ##
data = np.genfromtxt('seeds_dataset.txt', usecols=range(7))
labels = np.genfromtxt('seeds_dataset.txt', usecols=7)

## MODEL INSTANTIATION ##
GMM = GaussianMixtureModel(data, labels)
KM = KMeansModel(data, labels)

## RUN ##
GMM.run(subplot_ = plt.subplot(2, 1, 1))
KM.run(subplot_ = plt.subplot(2, 1, 2))




##KM.run(subplot_ = plt.subplot(2, 1, 2))

## SHOW PLOT ##
plt.show()
