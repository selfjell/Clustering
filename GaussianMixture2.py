import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import itertools
from scipy import linalg
import matplotlib as mpl


def create_dataset():
    df = pd.read_csv('seeds_dataset_CLEANED.csv', sep=' ', header=None,
                     names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "type"])
    f1 = df['f1'].values
    f2 = df['f2'].values
    f3 = df['f3'].values
    f4 = df['f4'].values
    f5 = df['f5'].values
    f6 = df['f6'].values
    f7 = df['f7'].values
    for_review = df['type'].values
    for_review = [x - 1 for x in for_review]
    X = np.matrix(list(zip(f1, f2, f3, f4, f5, f6, f7)))
    return X, for_review


def scale_data(X):
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled


def reduce_to_two_components(X_scaled):
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    print('Original shape: ', str(X_scaled.shape))
    print('Reduced shape: ', str(X_pca.shape))
    return X_pca


# Only works sometimes depending on the initial starting positions of kmeans.
def test_gmm(gmm_labels, testCluster):
    score = 0
    print(gmm_labels)
    print(testCluster)

    for labelX, labelY in zip(gmm_labels, testCluster):
        if labelX == labelY:
            score += 1

    # Percentage right given that the starting random seed corresponds to kmeans clustering groups
    score = (score / len(testCluster)) * 100
    print(score, '%')


#######
######
#####
######
######

def plot_results(X, Y_, means, covariances):
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                                  'darkorange'])
    splot = plt.subplot(111)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title("Gaussian Mixture Model")


X, testCluster = create_dataset()
X_Scaled = scale_data(X)
X_pca = reduce_to_two_components(X_Scaled)
X_pca_withoutScale = reduce_to_two_components(X)
OriginalX = X
X = X_pca

# Fit a Gaussian mixture with X_pca using three components
gmm = GaussianMixture(n_components=3, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_)
plt.show()

labels = gmm.predict(X)
test_gmm(labels, testCluster)

gmm = GaussianMixture(n_components=3, covariance_type='full').fit(OriginalX)
labels = gmm.predict(OriginalX)
test_gmm(labels, testCluster)

#gmm = GaussianMixture(n_components=3, covariance_type='full').fit(X_pca_withoutScale)
#labels = gmm.predict(X_Scaled)
#test_gmm(labels, testCluster)




f1 = np.array(X_pca[:, [0]]).ravel()
f2 = np.array(X_pca[:, [1]]).ravel()
plt.plot(f1, f2, 'bx')
plt.axis('equal')
plt.title('datapoints')
plt.show()

# viktig at det kjører raskt og effektivt. -Lab
# Kjøre koden, forkalre bra, og ha en kode som fungrer