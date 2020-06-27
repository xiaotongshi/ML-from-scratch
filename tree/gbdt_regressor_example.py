import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from gbdt import GBDTRegressor

def main():

    print ("-- Gradient Boosting Classification --")

    data = datasets.load_boston()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    # print(y_train)

    clf = GBDTRegressor()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = mean_squared_error(y_test, y_pred)
    print ("mse:", accuracy)


    # Plot().plot_in_2d(X_test, y_pred,
    #     title="Gradient Boosting",
    #     accuracy=accuracy,
    #     legend_labels=data.target_names)



if __name__ == "__main__":
    main()