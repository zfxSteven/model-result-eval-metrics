# *-* coding=utf-8 *-*
'''
    一些数学工具的实现
'''


# class metricsError(Exception):
#     '''异常类型'''
#     def __init__(self):
#         super(metricsError, self).__init__()
from scipy import integrate

class metrics(staticmethod):
    '''模型结果统计评估等'''
    epslion = 1e-5

    def __init__(self):
        super(metrics, self).__init__()
        pass

    @staticmethod
    def basic_check(y_true: list, y_pred: list, ):
        '''
        简单的数据检查
        :return:
        '''
        if len(y_true) != len(y_pred):
            raise Exception(f"序列长度不相同 y_true:{len(y_true)}, y_pred:{len(y_pred)}")

    @staticmethod
    def tp(y_true: list, y_pred: list, pos_label=1):
        '''
        True Positive
        Reference:
            1. https://en.wikipedia.org/wiki/Precision_and_recall
            2. https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF
        '''
        metrics.basic_check(y_true, y_pred, )
        return sum([int(y_true[idx] == pos_label and y_pred[idx] == pos_label) for idx, var in enumerate(y_pred)])

    @staticmethod
    def tn(y_true: list, y_pred: list, pos_label=1):
        '''
        True negative
        Reference:
            1. https://en.wikipedia.org/wiki/Precision_and_recall
            2. https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF
        '''
        metrics.basic_check(y_true, y_pred, )
        return sum([int(y_true[idx] != pos_label and y_pred[idx] != pos_label) for idx, var in enumerate(y_pred)])

    @staticmethod
    def fp(y_true: list, y_pred: list, pos_label=1):
        '''
        False Positive
        Reference:
            1. https://en.wikipedia.org/wiki/Precision_and_recall
            2. https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF
        '''
        metrics.basic_check(y_true, y_pred, )
        return sum([int(y_true[idx] != pos_label and y_pred[idx] == pos_label) for idx, var in enumerate(y_pred)])

    @staticmethod
    def fn(y_true: list, y_pred: list, pos_label=1):
        '''
        False negative
        Reference:
            1. https://en.wikipedia.org/wiki/Precision_and_recall
            2. https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF
        '''
        metrics.basic_check(y_true, y_pred, )
        return sum([int(y_true[idx] == pos_label and y_pred[idx] != pos_label) for idx, var in enumerate(y_pred)])

    @staticmethod
    def precision(y_true: list, y_pred: list, pos_label=1):
        '''
        tp / ( tp + fp + epslion)
        预测为阳性中真阳性的比例
        Reference: https://en.wikipedia.org/wiki/Precision_and_recall
        :return:
        '''
        metrics.basic_check(y_true, y_pred, )
        tp = metrics.tp(y_true, y_pred, pos_label)
        fp = metrics.fp(y_true, y_pred, pos_label)
        return 1.0 * tp / (tp + fp + metrics.epslion)

    @staticmethod
    def recall(y_true: list, y_pred: list, pos_label=1):
        '''
        tp / ( tp + fn + epslion)
        真阳性率（TPR）
        Recall in this context is also referred to as the true positive rate or sensitivity
        Reference: https://en.wikipedia.org/wiki/Precision_and_recall
        :return:
        '''
        metrics.basic_check(y_true, y_pred, )
        tp = metrics.tp(y_true, y_pred, pos_label)
        fn = metrics.fn(y_true, y_pred, pos_label)
        return 1.0 * tp / (tp + fn + metrics.epslion)

    @staticmethod
    def f1_score(y_true: list, y_pred: list, pos_label=1):
        '''
        2 * precision * recall / (precision + recall + epslion)
        :param y_true:
        :param y_pred:
        :param pos_label:
        :return:
        '''
        metrics.basic_check(y_true, y_pred, )
        p = metrics.precision(y_true, y_pred, pos_label)
        r = metrics.recall(y_true, y_pred, pos_label)
        return 2.0 * p * r / (p + r + metrics.epslion)

    @staticmethod
    def tpr(y_true: list, y_pred: list, pos_label=1):
        '''
        见 recall
        Reference:
            1. https://en.wikipedia.org/wiki/Precision_and_recall
            2. https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF
        :return:
        '''
        metrics.basic_check(y_true, y_pred, )
        return metrics.recall(y_true, y_pred, pos_label)

    @staticmethod
    def fpr(y_true: list, y_pred: list, pos_label=1):
        '''
        fp / ( fp + tn + epslion)
        伪阳性率（FPR）
        Reference: https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF
        :return:
        '''
        metrics.basic_check(y_true, y_pred, )
        fp = metrics.fp(y_true, y_pred, pos_label)
        tn = metrics.tn(y_true, y_pred, pos_label)
        return fp / ( fp + tn + metrics.epslion)

    @staticmethod
    def roc_curve(y_true: list, y_pred_score: list, pos_label=1, thresholds:list=None):
        '''
        ROC曲线
        sklearn中的ROC曲线点较少, 原因是阈值选取的较少， 不太容易绘制一个比较平滑的ROC曲线
        这里的thresholds默认是用score的四分点来计算，也可以自己指定
        默认negative label 为 0
        Reference: https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF
        :param y_true: 真实标签
        :param y_pred_score: 预测分数（一般为[0,1]的置信度）
        :param thresholds: 这里的thresholds默认是用score的四分点来计算，也可以自己指定
        :return:
            ROC-x-axis: fpr_lis
            ROC-y-axis: tpr_lis
            thresholds
        '''
        metrics.basic_check(y_true, y_pred_score,)
        neg_label = 0

        if thresholds is None:
            quarter = max(y_pred_score) - min(y_pred_score)
            thresholds = [ quarter*i for i in range(5)]

        fpr_lis = []
        tpr_lis = []

        def active_func(_score, _threshold, _pos_label, _neg_label):
            if _score >= _threshold:
                return _pos_label
            else:
                return _neg_label

        for threshold in thresholds:
            y_pred = [ active_func(score, threshold, pos_label, neg_label) for score in y_pred_score ]
            fpr_lis.append(metrics.fpr(y_true, y_pred, pos_label))
            tpr_lis.append(metrics.tpr(y_true, y_pred, pos_label))

        return fpr_lis, tpr_lis, thresholds

    @staticmethod
    def under_curve_area(x, y):
        '''基于点集的曲线积分'''
        return integrate.trapz(y, x)

    @staticmethod
    def auc(fpr_lis:list, tpr_lis:list):
        '''
        AUC: ROC曲线下面积
        Reference: https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF
        :param fpr_lis:
        :param tpr_lis:
        :return:
        '''
        return metrics.under_curve_area(fpr_lis, tpr_lis)


if __name__ == '__main__':
    import random
    sample_num = 100
    label_lis = [ random.randint(0,1) for _ in range(sample_num)]
    pred_lis = [ random.randint(0,1) for _ in range(sample_num)]
    pred_score_lis = [ random.uniform(0,1) for _ in range(sample_num)]

    print('-'*100)
    print(f"tp:{metrics.tp(label_lis, pred_lis, pos_label=1)}")
    print(f"fp:{metrics.fp(label_lis, pred_lis, pos_label=1)}")
    print(f"tn:{metrics.tn(label_lis, pred_lis, pos_label=1)}")
    print(f"fn:{metrics.fn(label_lis, pred_lis, pos_label=1)}")

    print(f"precision:{metrics.precision(label_lis, pred_lis, pos_label=1)}")
    print(f"recall:{metrics.recall(label_lis, pred_lis, pos_label=1)}")
    print(f"tpr:{metrics.tpr(label_lis, pred_lis, pos_label=1)}")
    print(f"fpr:{metrics.fpr(label_lis, pred_lis, pos_label=1)}")

    print(f"f1_score:{metrics.f1_score(label_lis, pred_lis, pos_label=1)}")

    print(f"roc_curve:")
    score_lis = [float(i)/100  for i in range(100)]
    fpr_lis, tpr_lis, thresholds = metrics.roc_curve(label_lis, pred_score_lis, thresholds=score_lis ,pos_label=1)
    # sort
    new_idx_lis = sorted(list(range(len(fpr_lis))), key=lambda idx:fpr_lis[idx], reverse=False)
    fpr_lis = [ fpr_lis[idx] for idx in new_idx_lis]
    tpr_lis = [ tpr_lis[idx] for idx in new_idx_lis]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr_lis, tpr_lis, '-', label=f'ROC curve(area={metrics.auc(fpr_lis, tpr_lis, )})')
    plt.plot(score_lis, score_lis, '--', color='red')
    # plt.plot(score_lis, p_score_lis, '--', color='yellow', label='thresholds')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

