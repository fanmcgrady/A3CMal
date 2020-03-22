import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize
from sklearn.metrics import confusion_matrix


def draw_cm(original, predicted, classifier_name):
    cm = confusion_matrix(original, predicted)
    names = np.unique(original)
    plot_confusion_matrix_fancy(cm, title='Confusion matrix on MicMalChal ' + classifier_name, names=names)


def plot_confusion_matrix_fancy(conf_arr, title='Confusion matrix', names=[]):
    norm_conf = []
    for i in conf_arr:
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure(figsize(10, 10))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues,
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.title(title)
    fig.colorbar(res)
    plt.xticks(range(width), names, rotation='vertical')
    plt.yticks(range(height), names)
    plt.savefig(title + '.png', format='png', dpi=200)
