from numpy.core.fromnumeric import reshape
from numpy.lib.function_base import append
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_boston(return_X_y=True)
trainx, testx, trainy, testy = train_test_split(X,
                                                y,
                                                test_size=0.33,
                                                random_state=42)
print(trainx.shape, testx.shape, trainy.shape, testy.shape)

# reconstruct trainx, trainy
aped = np.ones(339).reshape(339, 1)
trainx_r = np.append(trainx, aped, axis=1)
trainy.reshape(339, 1)

# construct A, B,
A = np.zeros(182).reshape(13, 14)
for i in range(13):
    A[i][i] = 1
B = np.zeros(14).reshape(1, 14)
B[0][13] = 1
# print(A, B)

# linear regression
beta1 = np.dot(np.dot(np.linalg.inv(np.dot(trainx_r.T, trainx_r)), trainx_r.T),
               trainy)
beta1.reshape(14, 1)
w1 = np.dot(A, beta1)
b1 = np.dot(B, beta1)
l1 = np.ones(167).reshape(167, 1)
predy1 = np.dot(testx, w1) + np.dot(l1, b1)
mse = 0
for i in range(167):
    mse = mse + (testy[i] - predy1[i])**2
mse = mse / 167
print(mse)

# ridge regression
I = np.identity(14)
one = np.ones(339).reshape(339, 1)
for lamda in range(1, 100):
    w_l = np.linalg.inv(
        339 * (np.dot(trainx.T, trainx) + 2 * lamda * np.identity(13)) -
        np.dot(np.dot(np.dot(trainx.T, one), one.T), trainx))
    w_r = 339 * np.dot(trainx.T, trainy) - np.dot(
        np.dot(np.dot(trainx.T, one), one.T), trainy)
    w2 = np.dot(w_l, w_r)
    b2 = (np.dot(one.T, trainy) - np.dot(np.dot(one.T, trainx), w2)) / 339

    # mse on test
    predy2 = np.dot(testx, w2) + np.dot(l1, b2)
    mse = 0
    for i in range(167):
        mse = mse + (testy[i] - predy2[i])**2
    mse = mse / 167
    print(mse)

    # mse on train
    predy2 = np.dot(trainx, w2) + np.dot(one, b2)
    mse = 0
    for i in range(167):
        mse = mse + (trainy[i] - predy2[i])**2
    mse = mse / 167
    print(mse)
