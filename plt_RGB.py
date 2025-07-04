import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

data = loadmat("img1.mat")
hsi_cube = data['ref'] 

rgb = hsi_cube[:, :, [29, 19, 9]]

rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))

plt.imshow(rgb)
plt.title("HSI RGB Composite")
plt.axis("off")
plt.show()