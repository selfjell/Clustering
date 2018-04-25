from GaussianMixtureModel import GaussianMixtureModel
import numpy as np
import matplotlib.pyplot as plt

## INIT ##
np.random.seed(18)

## DATA IMPORT ##
data = np.genfromtxt('seeds_dataset.txt', usecols=range(7))
labels = np.genfromtxt('seeds_dataset.txt', usecols=7)

## MODEL INSTANTIATION ##
GMM = GaussianMixtureModel(data, labels)
GMM.run()

## SHOW PLOT ##
plt.show()
