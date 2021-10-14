import numpy as np

data = np.load('results_2/00000.npz')

print(data['cdfs'].shape, data['zcdfs'].shape)