import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

class GaussianMixtureModel():

    def __init__(self, data, labels):
        self.data = data
        self.processed_data = data
        self.labels = labels
        self.operationsOrder = []

    def reduce(self, n_components = 2):
        self.operationsOrder.append("reduce")
        pca = decomposition.PCA(n_components = n_components)
        pca.fit(self.data)
        self.processed_data = pca.transform(self.processed_data)
        self.pca = pca

    def scale(self, scaling_type):
        self.operationsOrder.append("scale")
        if(scaling_type == "standard"):
            self.scaler = preprocessing.StandardScaler().fit(self.processed_data)
        elif(scaling_type == "normalizer"):
            self.scaler = preprocessing.Normalizer().fit(self.processed_data)
        elif(scaling_type == "minmax"):
            self.scaler = preprocessing.MinMaxScaler().fit(self.processed_data)
        elif(scaling_type == "robust"):
            self.scaler = preprocessing.RobustScaler().fit(self.processed_data)
        else:
            print("Illegal Argument given for scaling.")

        self.processed_data = self.scaler.transform(self.processed_data)

    def fit(self, n_components=3 , covariance_type = "full"):
        self.model = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
        self.model.fit(self.processed_data)

    def plot(self, colors = ['cyan', 'green', 'red'], subplot_ = plt.subplot(111)):
        for n, color in enumerate(colors):
            if self.model.covariance_type == 'full':
                covariances = self.model.covariances_[n][:2, :2]
            elif self.model.covariance_type == 'tied':
                covariances = self.model.covariances_[:2, :2]
            elif self.model.covariance_type == 'diag':
                covariances = np.diag(self.model.covariances_[n][:2])
            elif self.model.covariance_type == 'spherical':
                covariances = np.eye(self.model.means_.shape[1]) * gmm.covariances_[n]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ellipse = mpl.patches.Ellipse(self.model.means_[n, :2], v[0], v[1], 180 + angle, color=color)
            ellipse.set_clip_box(subplot_.bbox)
            ellipse.set_alpha(0.3)
            subplot_.add_artist(ellipse)
            ## ------------------------------
            if self.operationsOrder[0] == "reduce":
                temp_data = self.pca.transform(self.data[self.labels==n+1])
                temp_data = self.scaler.transform(temp_data)
            else:
                temp_data = self.scaler.transform(self.data[self.labels==n+1])
                temp_data = self.pca.transform(temp_data)
            plt.scatter(temp_data[:, 0], temp_data[:, 1], color=color)

    def run(self, plot = True, subplot_ = plt.subplot(111), test = True, reduce_first = True, reduce_components = 2, scaling_type="standard", model_components = 3, covariance_type="full"):
        if(reduce_first):
            self.reduce(reduce_components)
            self.scale(scaling_type)
        else:
            self.scale(scaling_type)
            self.reduce(reduce_components)
        self.fit(model_components, covariance_type = covariance_type)
        if plot: self.plot(subplot_ = subplot_)
        if test: self.test()

    def test(self):
        temp_labels = []
        for label in self.labels:
            temp_labels.append(label-1)
        self.labels = temp_labels
        score = 0
        for labelX, labelY in zip(self.model.predict(self.processed_data), self.labels):
            if labelX == labelY:
                score += 1
        # Percentage right given that the starting random seed corresponds to kmeans clustering groups
        score = (score / len(self.labels)) * 100
        print("Score: {}%".format(score))
        return score

    def reset(self):
        self.processed_data = self.data
