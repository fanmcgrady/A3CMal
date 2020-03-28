from tools.features.generate_novel import *
from tools.features.generate_winner import *

class FeatureExtract():
    def __init__(self, model):
        self.model = model

    # predict后直接可以调用，返回state
    def get_state(self, bytez):
        if 'winner' in self.model:
            generator = GenerateWinnerFeature(bytez)
        else:
            generator = GenerateNovelFeature(bytez)

        return generator.get_features()