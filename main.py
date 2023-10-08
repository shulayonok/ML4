from funcs import *
import numpy as np
import matplotlib.pyplot as plt

counter = lambda List, isPositive: sum(List) if isPositive else N - sum(List)

# Выборка
N = 1000

# Class 0
mu_0 = 190
sigma_0 = 10
basketballers = np.random.normal(mu_0, sigma_0, N)

# Class 1
mu_1 = 173
sigma_1 = 12
footballers = np.random.normal(mu_1, sigma_1, N)

# Порог бинарного классификатора
limit = 250

tpr_list, fpr_list, alpha_list, one_minus_beta, accuracy_list = [], [], [], [], []
TruePositive, FalsePositive, FalseNegative, TrueNegative = [], [], [], []

for lim in range(limit):
    classified_basketballers = Classification(basketballers, lim)
    classified_footballers = Classification(footballers, lim)

    TruePositive.append(counter(classified_footballers, True))
    TrueNegative.append(counter(classified_basketballers, False))
    FalsePositive.append(counter(classified_basketballers, True))
    FalseNegative.append(counter(classified_footballers, False))

    if TruePositive[lim] != 0 and (FalsePositive[lim] + TrueNegative[lim]) != 0:
        accuracy = Accuracy(TruePositive[lim], TrueNegative[lim], FalsePositive[lim], FalseNegative[lim])
        accuracy_list.append(accuracy)

        alpha = ALPHA(FalsePositive[lim], TrueNegative[lim])
        alpha_list.append(alpha)

        beta = BETA(FalseNegative[lim], TruePositive[lim])
        one_minus_beta.append(1 - beta)

        precision = Precision(TruePositive[lim], FalsePositive[lim])

        recall = Recall(TruePositive[lim], FalseNegative[lim])

        f1 = F1(precision, recall)

        tpr_list.append(TPR(TruePositive[lim], FalseNegative[lim]))

        fpr_list.append(FPR(FalsePositive[lim], TrueNegative[lim]))
    else:
        accuracy_list.append(0)

# Площадь под под построенной кривой (AUC)
print(f"AUC: {S(fpr_list, tpr_list)}")

# Максимальный Accuracy
index = accuracy_list.index(max(accuracy_list))
print(f"Порог при максимальном значении Accuracy: {index}")

TP_max = TruePositive[index]
FP_max = FalsePositive[index]
FN_max = FalseNegative[index]
TN_max = TrueNegative[index]

accuracy = Accuracy(TP_max, TN_max, FP_max, FN_max)
print(f"Accuracy = {accuracy}")

precision = Precision(TP_max, FP_max)
print(f"Precision = {precision}")

recall = Recall(TP_max, FN_max)
print(f"Recall = {recall}")

f1 = F1(precision, recall)
print(f"F1-score = {f1}")

alpha = ALPHA(FP_max, TN_max)
print(f"Alpha = {alpha}")

beta = BETA(FN_max, TP_max)
print(f"Beta = {beta}")

# График ROC кривой
fig = plt.figure()
plt.plot(alpha_list, one_minus_beta)
plt.show()
