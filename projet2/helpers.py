import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def convertDir2Deg(d):
    if d == 'N':
        return 0
    elif d == 'NNE':
        return 22.5
    elif d == 'NE':
        return 45
    elif d == 'ENE':
        return 67.5
    elif d == 'E':
        return 90
    elif d == 'ESE':
        return 112.5
    elif d == 'SE':
        return 135
    elif d == 'SSE':
        return 157.5
    elif d == 'S':
        return 180
    elif d == 'SSW':
        return 202.5
    elif d == 'SW':
        return 225
    elif d == 'WSW':
        return 247.5
    elif d == 'W':
        return 270.0
    elif d == 'WNW':
        return 292.5
    elif d == 'NW':
        return 315
    elif d == 'NNW':
        return 337.5
    else:
        raise