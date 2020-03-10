import glob
import os

import numpy as np

pe = '../../Dataset/train'

files = glob.glob(pe)

length = []
for fp in glob.glob(os.path.join(pe, '*.bytes')):
    with open(fp, 'r') as f:
        length.append(len(f.readlines()))

print(min(length))
print(max(length))
print(np.mean(length))