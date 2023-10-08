def ALPHA(FP, TN):
    return FP / (FP + TN)


def BETA(FN, TP):
    return FN / (FN + TP)


def Accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FN + FP)


def Precision(TP, FP):
    return TP / (TP + FP)


def Recall(TP, FN):
    return TP / (TP + FN)


def F1(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def FPR(FP, TN):
    return FP / (TN + FP)


def TPR(TP, FN):
    return TP / (TP + FN)


def Classification(data, t):
    classificated = []
    for height in data:
        if height > t:
            classificated.append(0)
        else:
            classificated.append(1)
    return classificated


def S(x, y):
    result = 0
    for i in range(len(x) - 1):
        result += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
    return result