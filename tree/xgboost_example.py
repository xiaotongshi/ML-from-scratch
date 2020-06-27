import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBoost

def main():
    print ("-- XGBoost --")

    data = datasets.load_boston()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    model = XGBoost()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # y_pred_line = model.predict(X)
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean Squared Error:", mse)

    # print(y_test[0:5])
    # # Color map
    # cmap = plt.get_cmap('viridis')



    # # Plot the results
    # m1 = plt.scatter(366 * X_train[:, 1], y_train, color=cmap(0.9), s=10)
    # m2 = plt.scatter(366 * X_test[:, 1], y_test, color=cmap(0.5), s=10)
    # m3 = plt.scatter(366 * X_test[:, 1], y_pred, color='black', s=10)
    # plt.suptitle("Regression Tree")
    # plt.title("MSE: %.2f" % mse, fontsize=10)
    # plt.xlabel('Day')
    # plt.ylabel('Temperature in Celcius')
    # plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    # plt.show()


if __name__ == "__main__":
    main()