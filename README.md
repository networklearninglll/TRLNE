# TRLNE

A TensorFlow Implementation of TRLNE(Transformer-based, Reinforcement Learning-refined approach to Network Ebmedding Learning)


# Quantify the noise of the path
In our experiments, to verify our hypothesis, we quantify the noise of the path as the mean degree of nodes in the path.

For classfication task on wiki dataset, we keep 22,350 path and abandon 1141 path, then we compare the mean degree of the selected paths and unselected paths.

The mean degree of the unselected paths is 4% higher than the selected paths.
is selected path |  path num | average degree
---|---
selected | 22350 | 150
unselected | 1141 | 156
