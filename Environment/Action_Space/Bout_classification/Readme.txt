From Joao Marques:
------------------

1) The kinematic parameters are in the "BoutKinematicParametersFinalArray" array. It has 240 columns. We used 74 of these to create the PCA space and to cluster the bout types. 

2) The "EnumeratorBoutKinPar.m" enumerator tells which column corresponds to which kinematic parameter.

3) The bout categorization is in the "BoutInfFinalArray". This array has information about the bouts that are not kinematic parameters. It also has its enumerator, called "EnumeratorBoutInf.m". The bout categorization is in column 134.

4) Each bout belongs to one of 13 types (1-13). Since I've done different categorizations over the years I use a vector that reorders the bout types always in the same order (idx = finalClustering.idx). The colors I use in the paper for each bout types are in: col = finalClustering.col follow the same order as in "idx". Anyway, for this categorization the ordering of the bouts is:
AS - 11
Slow1 - 7
Slow2 - 9
Short capture swim- 1
Long capture swim-2
Burst swim - 3
J-turn -5
HAT - 13
RT -  8
SAT - 12
O-bend - 4
LLC - 10
SLC - 6
 So the bouts that have 11 in the  BoutInfFinalArray(:,134) were categorized as approach swims. 