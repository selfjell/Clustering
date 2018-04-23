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


def run(plotting_, covar, ordered, scaling, seed = 18, index = 1, n_estimators = 1):
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
        plotting(gmm,data,ordered,pca,labels,scaler, index = index, n_estimators = n_estimators)

    lab = gmm.predict(data_red)
    print(test_gmm2(lab,labels))


## PLOT ##
def plotting(gmm,data,ordered,pca,labels,scaler, index = 1, n_estimators = 1):
    colors = ['cyan', 'green', 'red']

    if n_estimators==1:
        sub = plt.subplot(111)
    else:
        sub = plt.subplot(2, n_estimators // 2, index + 1)

    for n, color in enumerate(colors):
        ## Code mostly from scikit's examples drawing
        ## elipses on using iris data set
        ## ------------------------------

        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
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

    #plt.show()



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


# New methods for SUPER_TEST
def run2(scaler, seed, pca, ordered, gm_type):
    data = np.genfromtxt('seeds_dataset.txt', usecols=range(7))
    labels_dataset = np.genfromtxt('seeds_dataset.txt', usecols=7)

    X_scaled = scale(scaler, data).transform(data)
    if ordered:
        X_pca, unused = reduce(X_scaled)
    else:
        reduced_data, unused2 = reduce(data)
        X_pca = scale(scaler, reduced_data).transform(reduced_data)

    np.random.seed(seed)
    if pca:
        gmm = GaussianMixture(n_components=3, covariance_type=gm_type)
        gmm.fit(X_pca)
        labels_gmm = gmm.predict(X_pca)
    else:
        gmm = GaussianMixture(n_components=3, covariance_type=gm_type)
        gmm.fit(X_scaled)
        labels_gmm = gmm.predict(X_scaled)

    return test_gmm2(labels_gmm, labels_dataset)


def test_gmm2(labels_gmm, labels_dataset):
    lab = []
    for label in labels_dataset:
        lab.append(label-1)
    score = 0
    labels_dataset = lab

    for labelX, labelY in zip(labels_gmm, labels_dataset):
        if labelX == labelY:
            score += 1

    # Percentage right given that the starting random seed corresponds to kmeans clustering groups
    score = (score / len(labels_dataset)) * 100
    return score


# new methods for in the "KMeans4-patch" that tests gmm with "n" components from 2-7 testing with TestEverything
def run3(scaler, seed, pca, ordered, gm_type):
    data = np.genfromtxt('seeds_dataset.txt', usecols=range(7))
    labels_dataset = np.genfromtxt('seeds_dataset.txt', usecols=7)

    X_scaled = scale(scaler, data).transform(data)
    if ordered:
        X_pca = reduce2(X_scaled, pca)
    else:
        reduced_data = reduce2(data, pca)
        X_pca = scale(scaler, reduced_data).transform(reduced_data)

    np.random.seed(seed)
    gmm = GaussianMixture(n_components=3, covariance_type=gm_type)
    gmm.fit(X_pca)
    labels_gmm = gmm.predict(X_pca)
    return test_gmm2(labels_gmm, labels_dataset)


def reduce2(data, n_components):
    pca = decomposition.PCA(n_components=n_components, )
    return pca.fit_transform(data)
