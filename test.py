from unittest.mock import patch
import numpy as np
import os

file_name = "VGG_16_4096d_features.npy"

path = os.path.join(os.getcwd(),'data','dataset','oxford5k',file_name)


with open(path,"rb") as f:

    data = np.load(f)

print(data.shape)