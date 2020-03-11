import glob
import os

import numpy as np
from tqdm import tqdm

pe = '../../Dataset/train'

length = []
files = glob.glob(os.path.join(pe, '*.bytes'))
for fp in tqdm(files):
    with open(fp, 'r') as f:
        length.append(len(f.readlines()))

print(min(length))
print(max(length))
print(np.mean(length))

