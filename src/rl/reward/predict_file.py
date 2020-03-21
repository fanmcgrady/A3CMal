import pickle

import xgboost as xgb

from tools.features.generate_novel import *
from tools.features.generate_winner import *

'''
预测
'''


class Predict():
    def __init__(self, model):
        self.model = model
        self.state = None
        # 载入模型，xgboost对象
        self.classifier = pickle.load(open(self.model, "rb"))

    def predict(self, bytez):
        # 提取特征
        if 'winner' in self.model:
            generator = GenerateWinnerFeature(bytez)
        else:
            generator = GenerateNovelFeature(bytez)

        self.state = generator.get_features()

        dtest = xgb.DMatrix(self.state, missing=-999)
        pred_class = self.classifier.predict(dtest)
        label = list(pred_class[0]).index(max(pred_class[0]))

        return label

    # predict后直接可以调用，返回state
    def get_state(self):
        return self.state

    # 不需要predict直接提取state
    def get_state_without_predict(self, bytez):
        if 'winner' in self.model:
            generator = GenerateWinnerFeature(bytez)
        else:
            generator = GenerateNovelFeature(bytez)

        self.state = generator.get_features()
        return self.state


# if __name__ == '__main__':
# predit = Predict(bytez, '../../Dataset/models/winner_model.dat')
# print(predit())
