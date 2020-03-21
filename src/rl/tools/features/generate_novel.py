import traceback

from tools.features.generate_feature import GenerateFeature
from byte_code_extraction_facade import *


class GenerateNovelFeature(GenerateFeature):
    def __init__(self, bytez):
        super().__init__(bytez)

    def get_features(self):

        # byte_entropy+byte_oneg+byte_str_lengths+byte_meta_data+byte_img1

        with open(self.file_name, 'r') as f:
            feature = []
            try:
                start_time = time.time()

                # Entropy特征
                entropy = byte_entropy(f)
                feature.extend(entropy)
                f.seek(0)

                # 提取1-gram特征
                oneg = byte_1gram(f)
                feature.extend(oneg)
                f.seek(0)

                # String_lengths特征
                str_lengths = byte_string_lengths(f)
                feature.extend(str_lengths)
                f.seek(0)

                # Meta data特征
                meta_data = byte_meta_data(self.file_name, f)
                feature.extend(meta_data)
                f.seek(0)

                # Images_1特征
                image1 = byte_image1(f)
                feature.extend(image1)
                f.seek(0)

                # 显示一个文件提取时间
                time_cost = time.time() - start_time
                print("Extraction feature cost time:{}".format(time_cost))

            except Exception as err:
                print(err, traceback.print_exc())
                print("Error", self.file_name)

        return np.array([feature])
