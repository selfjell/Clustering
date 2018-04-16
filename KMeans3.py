import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.preprocessing as pp
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
    for_review = [x - 1 for x in for_review]
    X = np.matrix(list(zip(f1, f2, f3, f4, f5, f6, f7)))
    return X, for_review


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


def scale_data(X, type_):
    if type_ == "standard":
        X_scaled = pp.StandardScaler().fit_transform(X)
    elif type_ == "robust":
        X_scaled = pp.RobustScaler().fit_transform(X)
    elif type_ == "minmax":
        X_scaled = pp.MinMaxScaler().fit_transform(X)
    elif type_ == "norm":
        X_scaled = pp.Normalizer().fit_transform(X)
    else:
        print("Illegal argument (scaling type)")
    return X_scaled


def reduce_to_two_components(X_scaled):
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    # print('Original shape: ', str(X_scaled.shape))
    # print('Reduced shape: ', str(X_pca.shape))
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


# Only works sometimes depending on the initial starting positions of kmeans.
def test_kmeans(kmeans, testCluster):
    kmeans_labels = kmeans.labels_
    score = 0
    print(kmeans_labels)
    print(testCluster)
    for labelX, labelY in zip(kmeans_labels, testCluster):
        if labelX == labelY:
            score += 1

    # Percentage right given that the starting random seed corresponds to kmeans clustering groups
    score = (score/len(testCluster))*100
    print(score, '%')


def run_everything(type_, ordered, seed):
    clean_text_file()
    X, testCluster = create_dataset()
    elbow_function(X)

    if not ordered:
        X_Scaled = scale_data(X, type_)
        X_pca = reduce_to_two_components(X_Scaled)
    else:
        X_pca = reduce_to_two_components(X)
        X_Scaled = scale_data(X_pca, type_)
        X_pca = X_Scaled

    elbow_function(X_pca)
    kmeans = make_kmeans(X_pca)
    kmeans_X = make_kmeans(X)
    plot_to_screen(kmeans, X_pca)

    # With 2 dimensions
    np.random.seed(seed)
    test_kmeans(kmeans, testCluster)

    # With 7 dimensions, (might need a different seed)
    np.random.seed(seed)
    test_kmeans(kmeans_X, testCluster)




# NEW methods for better testing!

def test_kmeans2(kmeans, testCluster):
    score = 0
    for labelX, labelY in zip(kmeans.labels_, testCluster):
        if labelX == labelY:
            score += 1
    score = (score/len(testCluster))*100
    return score


def run_everything2(scaler, seed, pca, ordered):
    clean_text_file()
    X, testCluster = create_dataset()
    if ordered:
        X_Scaled = scale_data(X, scaler)
        X_pca = reduce_to_two_components(X_Scaled)
    else:
        X_pca = reduce_to_two_components(X)
        X_Scaled = scale_data(X_pca, scaler)
        X_pca = X_Scaled

    if pca:
        kmeans = make_kmeans(X_pca)
    else:
        kmeans = make_kmeans(X)
    np.random.seed(seed)
    return test_kmeans2(kmeans, testCluster)