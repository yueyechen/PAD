# coding=utf-8
import os
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def make_folder_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def evaluate_on_gt_and_pred(scores, labels):
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    index = np.lexsort((labels, scores))
    scores = scores[index]
    labels = labels[index]

    pos = np.sum(labels)
    neg = labels.shape[0] - pos

    TPR = np.cumsum(labels[::-1])[::-1]
    FPR = np.cumsum((1 - labels)[::-1])[::-1]
    acc = TPR + (neg - FPR)
    acc = acc / (neg + pos)

    bestAcc = np.max(acc)
    bestThresh = np.where(acc == bestAcc)[0]
    print('number of bestThresh: %3d' % bestThresh.shape[0])
    if bestThresh.shape[0] > 1:
        bestThresh = bestThresh[-1]
    TPR_atBestThresh = TPR[bestThresh] / pos
    FPR_atBestThresh = FPR[bestThresh] / neg
    APCER = FPR[bestThresh] / neg
    NPCER = (pos - TPR[bestThresh]) / pos
    ACER = (APCER + NPCER) / 2
    bestThresh = scores[bestThresh]

    # compute APCER,NPCER,ACER when thresh=0.5
    thresh = 0.5
    predict_issame = np.greater(scores, thresh)
    tp = np.sum(np.logical_and(predict_issame, labels))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(labels)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(labels)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), labels))

    APCER = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    NPCER = 0 if (fn + tp == 0) else float(fn) / float(fn + tp)
    ACER = (APCER + NPCER) / 2

    pre_TPR = TPR / pos
    pre_FPR = FPR / neg
    # TPR@FPR=10e-2
    FPR_01 = np.where(pre_FPR >= 0.01)[0][-1]
    TPR_01 = pre_TPR[FPR_01]
    Thresh_at01 = scores[FPR_01]
    # TPR@FPR=10e-3
    FPR_001 = np.where(pre_FPR >= 0.001)[0][-1]
    TPR_001 = pre_TPR[FPR_001]
    Thresh_at001 = scores[FPR_001]
    # TPR@FPR=10e-4
    FPR_0001 = np.where(pre_FPR >= 0.0001)[0][-1]
    TPR_0001 = pre_TPR[FPR_0001]
    Thresh_at0001 = scores[FPR_0001]
    # TPR@FPR=10e-5
    FPR_00001 = np.where(pre_FPR >= 0.00001)[0][-1]
    TPR_00001 = pre_TPR[FPR_00001]
    Thresh_at00001 = scores[FPR_00001]

    return bestAcc, bestThresh, APCER, NPCER, ACER, TPR_01, Thresh_at01, TPR_001, Thresh_at001, TPR_0001, Thresh_at0001, TPR_00001, Thresh_at00001
