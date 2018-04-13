import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from sklearn import decomposition
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

## DIMENSIONALITY REDUCTION (PCA) ##
def reduce(data):
    pca = decomposition.PCA(n_components=2, )
    pca.fit(data)
    return pca.transform(data), pca

## SCALING ##
def scale(scaling, data_red):
    scaler = None
    if scaling == 'standard':
        scaler = preprocessing.StandardScaler().fit(data_red)
    elif scaling == 'norm':
        scaler = preprocessing.Normalizer().fit(data_red)
    elif scaling == 'minmax':
        scaler = preprocessing.MinMaxScaler().fit(data_red) #best score
    elif scaling == 'robust':
        scaler = preprocessing.RobustScaler().fit(data_red)
    else:
        print("illegal argument")
    return scaler


def run(plotting_, covar, ordered, scaling, seed = 18):
    data = np.genfromtxt('seeds_dataset.txt', usecols=range(7))
    labels = np.genfromtxt('seeds_dataset.txt', usecols=7)


    np.random.seed(seed)


    if ordered:
        data_red, pca = reduce(data)

        scaler = scale(scaling, data_red)
        data_red = scaler.transform(data_red)
    else:
        scaler = scale(scaling, data)
        data_temp = scaler.transform(data)

        data_red, pca = reduce(data_temp)

    ## GAUSSIAN MIXTURE ##
    gmm = GaussianMixture(n_components=3, covariance_type=covar)
    gmm.fit(data_red)

    if plotting_:
        plotting(gmm,data,ordered,pca,labels,scaler)

    lab = gmm.predict(data_red)
    test_gmm(lab,labels)


## PLOT ##
def plotting(gmm,data,ordered,pca,labels,scaler):
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
        if ordered:
            temp_data = pca.transform(data[labels==n+1])
            temp_data = scaler.transform(temp_data)
        else:
            temp_data = scaler.transform(data[labels==n+1])
            temp_data = pca.transform(temp_data)
        plt.scatter(temp_data[:, 0], temp_data[:, 1], color=color)

    plt.show()



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
