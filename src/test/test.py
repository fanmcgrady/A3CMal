import glob
import os
import pickle

mc_path = '../../Dataset/models/mc.dat'
pe = '../../Dataset/pe'
train = '../../Dataset/train'
label_file = '../../Dataset/trainLabels.csv'

mc_map = pickle.load(open(mc_path, 'rb'))

content = []
with open(label_file, 'r') as csv:
    content = csv.readlines()

del_pe_count = 0
del_bytes_count = 0
del_label_count = 0
for filename in mc_map.keys():
    # 删除pe目录样本
    fold = filename.split('.')[0].lower()
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
        filename2 = '"' + filename + '"'
        if cc.startswith(filename2):
            content.remove(cc)
            del_label_count += 1
            break

# 统计pe
pe_total = 0
for fold in os.listdir(pe):
    pe_total += len(os.listdir(os.path.join(pe, fold)))

# 统计bytes
bytes = glob.glob(os.path.join(train, '*.bytes'))

print("删除{}个，剩余{}个".format(del_pe_count, pe_total))
print("删除{}个bytes文件，剩余{}个".format(del_bytes_count, len(bytes)))
print("删除{}个label，trainLabel.csv剩余{}个"
      .format(del_label_count, len(content) - 1))

if pe_total == len(bytes) and len(bytes) == len(content) - 1:
    print("清理成功，样本数为：{}".format(pe_total))
else:
    print("清理失败，分别为：{}，{}，{}".format(pe_total, len(bytes), len(content) - 1))

with open(label_file, 'w') as csv:
    csv.writelines(content)
