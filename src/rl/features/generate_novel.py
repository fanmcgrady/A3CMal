from generate_feature import GenerateFeature

class GenerateNovelFeature(GenerateFeature):
    def __init__(self, path):
        super().__init__(path)

    def get_features(self):
        # byte_entropy+byte_oneg+byte_str_lengths+byte_meta_data+byte_img1

        # load_model
        pass