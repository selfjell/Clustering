import KMeans2 as km
import test as gm

gm.run(True,'full',True,'norm')
gm.run(True,'full',True,'minmax')
gm.run(True,'full',True,'standard')
gm.run(True,'full',True,'robust')

km.run_everything("minmax",True,18)
km.run_everything("robust",True,18)
km.run_everything("standard",True,18)
km.run_everything("norm",True,18)
