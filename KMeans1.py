import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def clean_text_file():
    s = open("seeds_dataset_CLEANED.csv", "w+")
    with open("seeds_dataset.csv", "r+") as f:
        for line in f:
            s.write(" ".join(line.split()) + "\n")


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
    for number in for_review:
        number -= 1
    X = np.matrix(list(zip(f1, f2, f3, f4, f5, f6, f7)))
    return X


def elbow_function(X):
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


def make_kmeans(X_pca):
    kmeans = KMeans(n_clusters=3).fit(X_pca)
    return kmeans


def plot_to_screen(kmeans, X_pca):
    x = X_pca[:, [0]].ravel()
    y = X_pca[:, [1]].ravel()
    cluster = kmeans.labels_
    centers = kmeans.cluster_centers_
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c=cluster)
    for i, j in centers:
        ax.scatter(i, j, c='red', marker='*')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.show()


def run_everything():
    clean_text_file()
    X = create_dataset()
    elbow_function(X)
    X_Scaled = scale_data(X)
    X_pca = reduce_to_two_components(X_Scaled)
    kmeans = make_kmeans(X_pca)
    plot_to_screen(kmeans, X_pca)


run_everything()
