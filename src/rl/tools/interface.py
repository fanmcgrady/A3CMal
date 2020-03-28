import csv
import glob
import sys
# add rl
from tqdm import tqdm

sys.path.append('../../')
# add winner
sys.path.append('../../kaggle_Microsoft_malware_full/')
# add novel
sys.path.append('../../novel_feature')

from reward.predict_file import *
module_path = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))

WINNER_MODEL = os.path.join(module_path, '../../../Dataset/models/winner_model.dat')
NOVEL_MODEL = os.path.join(module_path, '../../../Dataset/models/novel_model.dat')

from tools.plot_cm import *

class Interface():
    def __init__(self, test=False):
        self.model = WINNER_MODEL
        # self.model = NOVEL_MODEL
        self.test = test
        self.predict = Predict(self.model)

    # 获取文件二进制数据
    def fetch_file(self, sha256):
        root = 'test' if self.test else 'train'
        root = os.path.join(module_path, '../../../Dataset/pe/' + root)

        location = os.path.join(root, sha256)
        try:
            with open(location, 'rb') as infile:
                bytez = infile.read()
        except IOError:
            print("Unable to read sha256 from {}".format(location))

        return bytez


    # 在samples目录中读取样本，放入list返回
    def get_available_sha256(self):
        root = 'test' if self.test else 'train'
        root = os.path.join(module_path, '../../../Dataset/pe/' + root)
        sha256list = []
        for fp in glob.glob(os.path.join(root, '*')):
            fn = os.path.split(fp)[-1]
            sha256list.append(fn)
        assert len(sha256list) > 0, "no files found in {} with sha256 names".format(root)
        return sha256list

    # 获取分类器label
    def get_label_local(self, bytez):
        label = self.predict.predict(bytez)
        state = self.predict.get_state()
        # label要加1
        label += 1
        return str(label), state

    def get_state(self, bytez):
        return self.predict.get_state_without_predict(bytez)

    # 加载label字典
    def get_original_label(self):
        label_map = {}
        root = os.path.join(module_path, '../../../Dataset/trainLabels.csv')
        for row in csv.DictReader(open(root)):
            label_map[row['Id']] = row['Class']

        # print("加载标签字典：{}个".format(len(label_map.keys())))
        return label_map

    # 绘制cm
    def draw(self, cm_name):
        file_list = self.get_available_sha256()
        label_map = self.get_original_label()

        original = []
        predict = []
        for filepath in tqdm(file_list):
            file_name = os.path.split(filepath)[-1]
            original.append(label_map.get(file_name, 0))

            bytez = self.fetch_file(filepath)
            label, _ = self.get_label_local(bytez)
            predict.append(str(label))
        draw_cm(original, predict, cm_name)

    def draw_after_train(self, before, after, cm_name):
        original = []
        predict = []
        for key in before.keys():
            original.append(before.get(key))
            predict.append(after.get(key))

        print('original:{}'.format(original))
        print('predict:{}'.format(predict))
        draw_cm(original, predict, cm_name)


# if __name__ == '__main__':
#     interface = Interface()
#     interface.draw('train set')
