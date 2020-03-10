import glob
import os

import numpy as np

pe = '../../Dataset/train'

length = []
files = glob.glob(os.path.join(pe, '*.bytes'))
for i, fp in enumerate(files):
    with open(fp, 'r') as f:
        length.append(len(f.readlines()))
    print("progress {}".format((i + 1)/len(files)))

print(min(length))
print(max(length))
print(np.mean(length))