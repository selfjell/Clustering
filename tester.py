import KMeans3 as km
import test as gm
import matplotlib.pyplot as plt

#gittest
n_estimators = 4
gm.run(True,'spherical',False,'norm', index = 0, n_estimators = n_estimators)
gm.run(True,'tied',False,'norm', index = 1, n_estimators = n_estimators)
gm.run(True,'full',False,'norm', index = 2, n_estimators = n_estimators)
gm.run(True,'diag',False,'norm', index = 3, n_estimators = n_estimators)
plt.show()
"""
km.run_everything("minmax",True,18)
km.run_everything("robust",True,18)
km.run_everything("standard",True,18)
km.run_everything("norm",True,18)
"""
