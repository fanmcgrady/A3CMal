import os
import pickle

from tqdm import tqdm

mc_path = '../../Dataset/models/mc.dat'
pe = '../../Dataset/pe'
train = '../../Dataset/train'
label_file = '../../Dataset/trainLabels.csv'
label_file_new = '../../Dataset/trainLabels_new.csv'

mc_map = pickle.load(open(mc_path, 'rb'))

content = []
content_new = []
with open(label_file, 'r') as csv:
    content = csv.readlines()
    content_new = content

for filename, tuple in mc_map.items():
    # 删除pe目录样本
    fold_list = os.listdir(pe)
    for i, fold in enumerate(fold_list):
        files = os.listdir(os.path.join(pe, fold))
        print('processing', fold)
        for f in tqdm(files):
            if f == filename:
                os.remove(os.path.join(os.path.join(pe, fold), f))

    # 删除.bytes
    os.remove(os.path.join(train, filename + '.bytes'))

    # 重新构造trainLabel.csv
    for cc in content:
        if filename in cc:
            content_new.remove(cc)

with open(label_file_new, 'w') as csv:
    csv.writelines(content_new)
