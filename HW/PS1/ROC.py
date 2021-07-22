import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv", index_col=0)
data = data.sort_values('output', ascending=False)
target = data['output'].unique()
n = len(target)
TPR = np.zeros(n)
FPR = np.zeros(n)
for i in range(n):
    data['prediction'] = data['output'] >= target[i]
    TP = np.sum((data['label'] == 1) & (data['prediction'] == 1))
    FP = np.sum((data['label'] == 0) & (data['prediction'] == 1))
    TN = np.sum((data['label'] == 0) & (data['prediction'] == 0))
    FN = np.sum((data['label'] == 1) & (data['prediction'] == 0))
    TPR[i] = TP / (TP + FN)
    FPR[i] = FP / (FP + TN)
plt.plot(FPR, TPR)
plt.axis([0, 1, 0, 1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()
AUC = np.sum((FPR[1:] - FPR[:-1]) * (TPR[:-1] + TPR[1:])) / 2
print("AUC: ", AUC)