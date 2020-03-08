import sys
import traceback

import numpy

sys.path.append('../')
from novel_feature.settings import *
from novel_feature.feature_extraction import *

directory_name = os.path.join(DATASET_PATH, 'train') + '/'
files = os.listdir(directory_name)
files = numpy.sort(files)
byte_files = [i for i in files if i.endswith('.bytes')]

for t, fname in enumerate(byte_files):
    with open(directory_name + fname, 'r') as f:
        try:
            entropy = byte_entropy(f)
        except Exception as err:
            print(err, traceback.print_exc())
            print("Error", fname)
