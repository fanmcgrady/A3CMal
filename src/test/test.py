import sys
import traceback

import numpy

sys.path.append('../novel_feature')
from feature_extraction import *

directory_name = os.path.join('../../Dataset', 'train') + '/'
files = os.listdir(directory_name)
files = numpy.sort(files)
byte_files = [i for i in files if i.endswith('.bytes')]

for t, fname in enumerate(byte_files):
    with open(directory_name + fname, 'r') as f:
        try:
            entropy = byte_entropy(f)
            if (t + 1) % 100 == 0:
                print("processing", t)
        except Exception as err:
            print(err, traceback.print_exc())
            print("Error", fname)
