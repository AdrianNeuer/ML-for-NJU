from sklearn import model_selection
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from deepforest import CascadeForestRegressor
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # From https://deep-forest.readthedocs.io/en/latest/index.html
    '''
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model_deep = CascadeForestRegressor(n_estimators=2)
    model_random = RandomForestRegressor()
    model_deep.fit(X_train, y_train)
    model_random.fit(X_train, y_train)
    y_pred_deep = model_deep.predict(X_test)
    y_pred_random = model_random.predict(X_test)
    mse_deep = mean_squared_error(y_test, y_pred_deep)
    mse_random = mean_squared_error(y_test, y_pred_random)
    print("\nTesting deep_forest_MSE: {:.3f}".format(mse_deep))
    print("\nTesting random_forest_MSE: {:.3f}".format(mse_random))
    '''

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    x = np.array([1, 2, 3, 4, 5, 6])
    Y_pred = np.array([0, 0, 0, 0, 0, 0])
    for i in range(1, 7):
        model = CascadeForestRegressor(n_estimators=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        Y_pred[i - 1] = mse
    plt.plot(x, Y_pred)
    plt.axis([1, 6, 5, 10])
    plt.xlabel('n_estimators')
    plt.ylabel('mse of deep-forest')
    plt.title('mse Curve')
    plt.show()