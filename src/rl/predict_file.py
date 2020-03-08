import os
import pickle
from features.generate_novel import *
from features.generate_winner import *

# 提取特征
geneator = GenerateNovelFeature('')
# 这里应该返回一个dataframe对象
extracted_feature = geneator.get_features('java.bytes')


# 载入模型，xgboost对象
classifier = pickle.load((open("models/novel_model.dat", "rb")))
pred_class = classifier.predict(extracted_feature)
pred_prob = classifier.predict_proba(extracted_feature)
print(pred_class)
print(pred_prob)



