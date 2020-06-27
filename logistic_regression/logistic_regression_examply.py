import numpy as np
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import linear_model
from logistic_regression import LogisticRegression

def main():
    # Load dataset
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf1 = linear_model.LogisticRegression()
    clf1 = LogisticRegression()
    clf1.fit(X_train, y_train)
    y_pred = clf1.predict(X_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    accuracy = accuracy_score(y_test, y_pred)
    print("sklearn lr Accuracy:", accuracy)

    clf2 = LogisticRegression()
    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    accuracy = accuracy_score(y_test, y_pred)
    print("Our lr Accuracy:", accuracy)

    # # Reduce dimension to two using PCA and plot the results
    # Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)


if __name__ == "__main__":
    main()