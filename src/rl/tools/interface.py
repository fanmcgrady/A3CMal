import csv
import glob
import pickle
import sys

# add rl
from tqdm import tqdm

sys.path.append('../../')
# add winner
sys.path.append('../../kaggle_Microsoft_malware_full/')
# add novel
sys.path.append('../../novel_feature')

import xgboost as xgb

from state.state import FeatureExtract

import os

module_path = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))

from tools.plot_cm import *

WINNER_MODEL = os.path.join(module_path, '../../../Dataset/models/winner_model.dat')
NOVEL_MODEL = os.path.join(module_path, '../../../Dataset/models/novel_model.dat')

MODEL_NAME = WINNER_MODEL
MODEL_CLASSIFIER = pickle.load(open(MODEL_NAME, "rb"))

feature_extractor = FeatureExtract()

# 获取文件二进制数据
def fetch_file(sha256, test=False):
    root = 'test' if test else 'train'
    root = os.path.join(module_path, '../../../Dataset/pe/' + root)

    location = os.path.join(root, sha256)
    try:
        with open(location, 'rb') as infile:
            bytez = infile.read()
    except IOError:
        print("Unable to read sha256 from {}".format(location))

    return bytez


# 在samples目录中读取样本，放入list返回
def get_available_sha256(test=False):
    root = 'test' if test else 'train'
    root = os.path.join(module_path, '../../../Dataset/pe/' + root)
    sha256list = []
    for fp in glob.glob(os.path.join(root, '*')):
        fn = os.path.split(fp)[-1]
        sha256list.append(fn)
    assert len(sha256list) > 0, "no files found in {} with sha256 names".format(root)
    return sha256list


# 获取分类器label
def get_label_local(bytez):
    # 提取特征
    state = feature_extractor.get_state(bytez)

    dtest = xgb.DMatrix(state, missing=-999)
    pred_class = MODEL_CLASSIFIER.predict(dtest)
    label = list(pred_class[0]).index(max(pred_class[0]))
    # label要加1
    label += 1
    return str(label)


# 加载label字典
def get_original_label():
    label_map = {}
    root = os.path.join(module_path, '../../../Dataset/trainLabels.csv')
    for row in csv.DictReader(open(root)):
        label_map[row['Id']] = row['Class']

    # print("加载标签字典：{}个".format(len(label_map.keys())))
    return label_map


# 绘制cm
def draw(cm_name):
    file_list = get_available_sha256()
    label_map = get_original_label()

    original = []
    predict = []
    for filepath in tqdm(file_list):
        file_name = os.path.split(filepath)[-1]
        original.append(label_map.get(file_name, 0))

        bytez = fetch_file(filepath)
        label = get_label_local(bytez)
        predict.append(str(label))
    draw_cm(original, predict, cm_name)


def draw_after_train(before, after, cm_name):
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
