from scipy.io import loadmat
import numpy as np

data = loadmat("img1.mat")

print("ref type:", type(data['ref']))
print("ref shape:", np.shape(data['ref']))

print("lbl type:", type(data['lbl']))
print("lbl shape:", np.shape(data['lbl']))