import pickle
import os

MODEL_PATH = '../../Dataset/models'

# 载入winner的模型
def load_model(model_path, model_name):
    classifier = pickle.load(os.path.join(model_path, model_name) , 'rb')
    return classifier

if __name__ == '__main__':
    model_name = ''
    cls = load_model(MODEL_PATH, model_name)
    sample = xgb.DMatrix(test)
    print(cls.predict(sample))
