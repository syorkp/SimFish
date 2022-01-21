import numpy as np
import scipy.io
import matplotlib.pyplot as plt
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
  
Important indices (subtract 1?):
  - boutAngle = 10 Angle turned during tail motion (degrees, clockwise negative)
  - distBoutAngle = 11 Angle turned during bout including glide (degrees, clockwise negative)
  - boutMaxAngle = 12 Maximum angle turned during bout (degrees, absolute value)
  - boutDistanceX = 15 Distance advanced in the tail to head direction during tail motion (forward positive, mm)
  - boutDistanceY = 16 Distance moved in left right direction during tail motion (left positive, mm)

Column 18. Distance advanced in the tail to head direction including glide (forward positive, mm)
Column 19. Distance moved in left right direction including glide (left positive, mm)


Questions:
* Units?
* What is the distinction between max angle and angle
* What are dist vs bout
"""

mat = scipy.io.loadmat("bouts.mat")
bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
angles = bout_kinematic_parameters_final_array[:, 9]
dist_angles = bout_kinematic_parameters_final_array[:, 10]
max_angles = bout_kinematic_parameters_final_array[:, 11]
distance_x = bout_kinematic_parameters_final_array[:, 14]
distance_y = bout_kinematic_parameters_final_array[:, 15]

# plt.hist(max_angles, bins=1000)
# plt.show()
#
# plt.hist(distance_x, bins=1000)
# plt.show()
#
# plt.hist(distance_y, bins=1000)
# plt.show()

# Want actual distance moved - combination of both.
distance = (distance_x**2 + distance_y**2)**0.5
# Plot distance against angles in heatmap to get idea of distribution.
plt.scatter(np.absolute(angles), distance)
plt.xlabel("Angle")
plt.ylabel("Distance moved")
plt.show()

plt.scatter(np.absolute(angles), np.absolute(dist_angles))
plt.show()

x = True

