import pickle

import xgboost as xgb

from features.generate_novel import *
from features.generate_winner import *

'''
预测
'''
class Predict():
    def __init__(self, file, model):
        self.file = file
        self.model = model

    def __call__(self, *args, **kwargs):
        # 提取特征
        if 'winner' in self.model:
            generator = GenerateWinnerFeature(self.file)
        else:
            generator = GenerateNovelFeature(self.file)

        extracted_feature = generator.get_features()

        # 载入模型，xgboost对象
        classifier = pickle.load(open(self.model, "rb"))
        dtest = xgb.DMatrix(extracted_feature, missing=-999)
        pred_class = classifier.predict(dtest)
        label = list(pred_class[0]).index(max(pred_class[0]))

        return label

if __name__ == '__main__':
    predit = Predict('java.exe', '../../Dataset/models/winner_model.dat')
    print(predit())