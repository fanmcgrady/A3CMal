from pandas import read_csv
from novel_feature.settings import *
import os

class_lable = read_csv(TRAIN_ID_PATH, delimiter=',')
map = {}
for i in range(1, 10):
    map[i] = 0

filelist = os.listdir(SAMPLES_PATH)

for name in filelist:
    name = name.split(".")[0]

    for index, row in class_lable.iterrows():
        if row['Id'] == name:
            map[row['Class']] += 1
            break

count = 0
for i, j in map:
    if j != 0:
        count += j
print(count)
print(map)


