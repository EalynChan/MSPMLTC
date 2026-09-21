
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, label_ranking_loss, hamming_loss, f1_score

def mll_metrics(y_test, y_pred, y_score):
    scorce = []
    rl = label_ranking_loss(y_test, y_score)
    print('rl: %.4f' % rl, end=', ')
    hl = hamming_loss(y_test, y_pred)
    print('hl: %.4f' % hl, end=', ')
    ap = label_ranking_average_precision_score(y_test, y_score)
    print('ap: %.4f' % ap, end=', ')
    f1 = f1_score(y_test, y_pred, average='macro')
    print('f1: %.4f' % f1)

    scorce.append(ap)
    scorce.append(f1)
    scorce.append(rl)
    scorce.append(hl)
    return np.round(scorce, 4)