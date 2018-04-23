import KMeans3 as km
import test as gm

#gittest
gm.run(True,'spherical',False,'norm')
gm.run(True,'tied',False,'norm')
gm.run(True,'full',False,'norm')
gm.run(True,'diag',False,'norm')
"""
km.run_everything("minmax",True,18)
km.run_everything("robust",True,18)
km.run_everything("standard",True,18)
km.run_everything("norm",True,18)
"""

# False = scale_first
km.run_everything("norm", False, 16)
