import glob

from reward.predict_file import *
module_path = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))


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
    root = '../../../Dataset/pe/' + root
    sha256list = []
    for fp in glob.glob(os.path.join(root, '*')):
        fn = os.path.split(fp)[-1]
        sha256list.append(fn)
    assert len(sha256list) > 0, "no files found in {} with sha256 names".format(root)
    return sha256list


# 获取分类器label
def get_label_local(bytez, model):
    predit = Predict(model)
    label = predit.predict(bytez)
    state = predit.get_state()
    return label, state
