import glob
import os
import sys

module_path = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))

# for local model
from env import PEFeatureExtractor

# 获取文件二进制数据
def fetch_file(sha256, test=False):
    root = ''
    location = os.path.join(root, sha256)
    try:
        with open(location, 'rb') as infile:
            bytez = infile.read()
    except IOError:
        print("Unable to read sha256 from {}".format(location))

    return bytez


def fetch_evaded_file(sha256):
    location = ''
    try:
        with open(location, 'rb') as infile:
            bytez = infile.read()
    except IOError:
        print("Unable to read sha256 from {}".format(location))

    return bytez


# 在samples目录中读取样本，放入list返回
def get_available_sha256():
    sha256list = []
    for fp in glob.glob(os.path.join('SAMPLE_PATH', '*')):
        fn = os.path.split(fp)[-1]
        # result = re.match(r'^[0-9a-fA-F]{64}$', fn) # require filenames to be sha256
        # if result:
        #     sha256list.append(result.group(0))
        sha256list.append(fn)
    assert len(sha256list) > 0, "no files found in {} with sha256 names".format(SAMPLE_PATH)
    return sha256list


# 在test-samples目录中读取样本，放入list返回
def get_available_test_sha256():
    sha256list = []
    for fp in glob.glob(os.path.join('TEST_SAMPLE_PATH', '*')):
        fn = os.path.split(fp)[-1]
        # result = re.match(r'^[0-9a-fA-F]{64}$', fn) # require filenames to be sha256
        # if result:
        #     sha256list.append(result.group(0))
        sha256list.append(fn)
    assert len(sha256list) > 0, "no files found in {} with sha256 names".format(TEST_SAMPLE_PATH)
    return sha256list

# 获取分类器评分
def get_score_local(bytez):
    # extract features
    feature_extractor = PEFeatureExtractor()
    features = feature_extractor.extract(bytez)

    # query the model
    # score = local_model.predict_proba(features.reshape(1, -1))[
    #     0, -1]  # predict on single sample, get the malicious score
    # return score


# 获取分类器label
def get_label_local(bytez):
    # mimic black box by thresholding here
    score = get_score_local(bytez)
    # label = float(get_score_local(bytez) >= local_model_threshold)
    # print("score={} (hidden), label={}".format(score, label))
    # return label
