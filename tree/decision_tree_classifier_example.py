import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decision_tree import ClassificationTree
from sklearn import tree

def main():

    print ("-- Classification Tree --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = ClassificationTree(max_features=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Our DTClassifier Accuracy:", accuracy)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print ("sklearn DTClassifier Accuracy:", accuracy)

    # Plot().plot_in_2d(X_test, y_pred,
    #     title="Decision Tree",
    #     accuracy=accuracy,
    #     legend_labels=data.target_names)


if __name__ == "__main__":
    main()