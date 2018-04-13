import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from sklearn import decomposition
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

data = np.genfromtxt('seeds_dataset.txt', usecols=range(7))
labels = np.genfromtxt('seeds_dataset.txt', usecols=7)


np.random.seed(18)


## DIMENSIONALITY REDUCTION (PCA) ##
pca = decomposition.PCA(n_components=2, )
pca.fit(data)
data_red = pca.transform(data)


## SCALING ##
#scaler = preprocessing.StandardScaler().fit(data_red)
#scaler = preprocessing.Normalizer().fit(data_red)
scaler = preprocessing.MinMaxScaler().fit(data_red) #best score
#scaler = preprocessing.RobustScaler().fit(data_red)
data_red = scaler.transform(data_red)


## GAUSSIAN MIXTURE ##
gmm = GaussianMixture(n_components=3, covariance_type='tied')
gmm.fit(data_red)


## PLOT ##
def plotting(gmm,data):
    colors = ['cyan', 'green', 'red']

    for n, color in enumerate(colors):
        ## Code mostly from scikit's examples drawing
        ## elipses on using iris data set
        ## ------------------------------
        sub = plt.subplot(111)
        v, w = np.linalg.eigh(gmm.covariances_[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(sub.bbox)
        ell.set_alpha(0.3)
        sub.add_artist(ell)
        ## ------------------------------
        temp_data = pca.transform(data[labels==n+1])
        temp_data = scaler.transform(temp_data)
        plt.scatter(temp_data[:, 0], temp_data[:, 1], color=color)

    plt.show()

#plotting(gmm,data)

def test_gmm(gmm_labels, testCluster):
    lab = []
    for label in testCluster:
        lab.append(label-1)
    testCluster = lab
    score = 0
    print(gmm_labels)
    print(testCluster)

    for labelX, labelY in zip(gmm_labels, testCluster):
        if labelX == labelY:
            score += 1

    # Percentage right given that the starting random seed corresponds to kmeans clustering groups
    score = (score / len(testCluster)) * 100
    print(score, '%')

lab = gmm.predict(data_red)
test_gmm(lab,labels)
