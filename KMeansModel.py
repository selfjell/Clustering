import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.preprocessing as pp
from sklearn.decomposition import PCA


class KMeansModel():

    def __init__(self, data, labels):
        self.data = data
        self.processed_data = self.data
        self.labels = labels
        self.operation_order = []

    # Used for finding out how many clusters are optimal for a given dataset (X)
    # The optimal number of clusters is the point on the x-axis were the graph cuts off like an elbow
    def elbow_graph(X):
        nClusters = range(1, 15)
        score = elbow_variant1(X, nClusters)
        # score = elbow_variant2(X, nClusters)
        plt.plot(nClusters, score)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title('Elbow Curve: - We see that the elbow is approximatly at 3 clusters')
        plt.show()


    def elbow_variant1(X, nClusters):
        kmeans = [KMeans(n_clusters=i) for i in nClusters]
        score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
        return score


    def elbow_variant2(X, nClusters):
        distorsions = []
        kmeans = [KMeans(n_clusters=i).fit(X) for i in nClusters]
        [distorsions.append(k.inertia_) for k in kmeans]
        return distorsions


    def scale_data(self, scaling_type):
        self.operation_order.append("scale")
        if scaling_type == "standard":
            self.processed_data = pp.StandardScaler().fit_transform(self.processed_data)
        elif scaling_type == "robust":
            self.processed_data = pp.RobustScaler().fit_transform(self.processed_data)
        elif scaling_type == "minmax":
            self.processed_data = pp.MinMaxScaler().fit_transform(self.processed_data)
        elif scaling_type == "norm":
            self.processed_data = pp.Normalizer().fit_transform(self.processed_data)
        else:
            print("Illegal argument (scaling type)")


    def reduce(self, n_components_):
        self.operation_order.append("reduce")
        pca = PCA(n_components = n_components_)
        pca.fit(self.processed_data)
        self.processed_data = pca.transform(self.processed_data)
        # print('Original shape: ', str(X_scaled.shape))
        # print('Reduced shape: ', str(X_pca.shape))


    def plot(self, subplot_ = plt.subplot(111)):
        x = self.processed_data[:, [0]].ravel()
        y = self.processed_data[:, [1]].ravel()

        cluster = self.model.labels_
        centers = self.model.cluster_centers_
        plt.scatter(x, y, c=cluster)
        for i, j in centers:
            plt.scatter(i, j, c='red', marker='*')
        plt.xlabel('x')
        plt.ylabel('y')

    def fit(self, model_components):
        self.model = KMeans(n_clusters = model_components)
        self.model = self.model.fit(self.processed_data)

    def reset(self):
        self.processed_data = self.data

    def run(self, plot = True, subplot_ = plt.subplot(111), test = True, reduce_first = True, reduce_components = 2, scaling_type = "standard", model_components = 3):

        if reduce_first:
            self.reduce(reduce_components)
            self.scale_data(scaling_type)
        else:
            self.scale_data(scaling_type)
            self.reduce(reduce_components)

        self.fit(model_components)

        if plot: self.plot(subplot_ = subplot_)

        if test:
            score = self.test_kmeans()
            print("Score: {}%".format(score))

    # Tests how many of kmeans.labels_ are identical to the test-labels/ground-truth-labels from the dataset (testCluster)
    def test_kmeans(self):
        score = 0
        for labelX, labelY in zip(self.model.labels_, self.labels):
            if labelX == labelY:
                score += 1
        score = (score/len(self.labels))*100
        return score


    # new method for KMeans4 testing with TestEverything
    def reduce_components(X_scaled, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        # print('Original shape: ', str(X_scaled.shape))
        # print('Reduced shape: ', str(X_pca.shape))
        return X_pca
