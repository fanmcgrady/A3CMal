import glob
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

del_pe_count = 0
del_bytes_count = 0
del_label_count = 0
for filename in mc_map.keys():
    # 删除pe目录样本
    fold = filename.split('.')[0].capitalize()
    pp = os.path.join(os.path.join(pe, fold), filename)
    if os.path.exists(pp):
        os.remove(pp)
        del_pe_count += 1

    # 删除.bytes
    ff = os.path.join(train, filename + '.bytes')
    if os.path.exists(ff):
        os.remove(ff)
        del_bytes_count += 1

    # 重新构造trainLabel.csv
    for cc in content:
        if filename in cc:
            content_new.remove(cc)
            del_label_count += 1

bytes = glob.glob(os.path.join(train, '*.bytes'))

print("原有{}个pe样本，删除{}个，剩余{}个".format(len(content) - 1, del_pe_count, len(content) - 1 - del_pe_count))
print("删除{}个bytes文件，剩余{}个".format(del_bytes_count, len(bytes)))
print("原有{}个label，删除{}个label，content_new剩余{}个"
      .format(len(content) - 1, del_label_count, len(content_new) - 1))

with open(label_file, 'w') as csv:
    csv.writelines(content_new)
