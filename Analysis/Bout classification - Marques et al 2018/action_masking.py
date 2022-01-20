import numpy as np
import scipy.io

import h5py

# f = h5py.File('BoutMapCenters_kNN4_74Kins4dims_1.75Smooth_slow_3000_auto_4roc_merged11.mat', 'r')
# data = f.get('data/variable1')
# data = np.array(data) # For converting to a NumPy array

"""
1) The kinematic parameters are in the "BoutKinematicParametersFinalArray" array. It has 240 columns. We used 74 of 
these to create the PCA space and to cluster the bout types. 

2) The "EnumeratorBoutKinPar.m" enumerator tells which column corresponds to which kinematic parameter.

3) The bout categorization is in the "BoutInfFinalArray". This array has information about the bouts that are not 
kinematic parameters. It also has its enumerator, called "EnumeratorBoutInf.m". The bout categorization is in column 
134.

4) Each bout belongs to one of 13 types (1-13). Since I've done different categorizations over the years I use a vector 
that reorders the bout types always in the same order (idx = finalClustering.idx). The colors I use in the paper for 
each bout types are in: col = finalClustering.col follow the same order as in "idx". Anyway, for this categorization 
the ordering of the bouts is:


Important numbers:
  - 59390 - Bouts
  - 33
  - 49
  - 8203 - Number of bouts that are not kinematic parameters
  - 232
"""

mat = scipy.io.loadmat("bouts.mat")
x = True

