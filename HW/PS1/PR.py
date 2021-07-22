import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv", index_col=0)
data = data.sort_values('output', ascending=False)
target = data['output'].unique()
n = len(target)
P = np.zeros(n)
R = np.zeros(n)
for i in range(n):
    data['prediction'] = data['output'] >= target[i]
    TP = np.sum((data['label'] == 1) & (data['prediction'] == 1))
    FP = np.sum((data['label'] == 0) & (data['prediction'] == 1))
    TN = np.sum((data['label'] == 0) & (data['prediction'] == 0))
    FN = np.sum((data['label'] == 1) & (data['prediction'] == 0))
    P[i] = TP / (TP + FP)
    R[i] = TP / (TP + FN)
plt.plot(R, P)
plt.axis([0, 1, 0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P-R Curve')
plt.show()
