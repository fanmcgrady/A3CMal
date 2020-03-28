from tools.features.generate_novel import *
from tools.features.generate_winner import *
from tools.interface import MODEL_NAME

class FeatureExtract():
    def __init__(self):
        pass

    # predict后直接可以调用，返回state
    def get_state(self, bytez):
        if 'winner' in MODEL_NAME:
            generator = GenerateWinnerFeature(bytez)
        else:
            generator = GenerateNovelFeature(bytez)

        return generator.get_features()