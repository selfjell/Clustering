from sklearn.mixture import GaussianMixture
from pathlib import Path
import matplotlib.pyplot as plt

def get_features(line,set_):
    list_ = line.split("	")
    if list_[7] == set_:
        for i in range(7):
            list_[i] = float(list_[i])
    return list_[0:7]

def get_dataset(set_):
    path = Path("..").joinpath("seeds_dataset.txt")
    lines = path.open("r").readlines()
    data = []
    for line in lines:
        example = get_features(line,set_)
        data.append(example)
    return data

train = get_dataset("1")
test = get_dataset("3")
gm = GaussianMixture(n_components = 7)

gm.fit(X = train)

ar = gm.predict(X=test)
print(ar)
print("Size: {}".format(len(ar)))
plt.figure(figsize=(12,12))
plt.scatter(test)
