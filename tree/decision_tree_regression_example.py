import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from decision_tree import RegressionTree
from sklearn import tree

def main():

    print ("-- Regression Tree --")
    data = datasets.load_boston()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    model = RegressionTree()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print ("Our DTRegressor Mean Squared Error:", mse)

    model = tree.DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print ("sklearn DTRegressor Mean Squared Error:", mse)

    # y_pred_line = model.predict(X)
    # # Color map
    # cmap = plt.get_cmap('viridis')

    # # Plot the results
    # # Plot the results
    # m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    # m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    # m3 = plt.scatter(366 * X_test, y_pred, color='black', s=10)
    # plt.suptitle("Regression Tree")
    # plt.title("MSE: %.2f" % mse, fontsize=10)
    # plt.xlabel('Day')
    # plt.ylabel('Temperature in Celcius')
    # plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    # plt.show()


if __name__ == "__main__":
    main()