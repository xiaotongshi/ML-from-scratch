import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random_forest import RandomForest
from sklearn import ensemble

def main():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print("X_train.shape:", X_train.shape)
    print("Y_train.shape:", y_train.shape)

    n_features = X_train.shape[1]
    clf = RandomForest(n_estimators=100, max_features=int(np.sqrt(n_features)))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Our RF Accuracy:", accuracy)

    clf = ensemble.RandomForestClassifier(n_estimators=100, max_features=int(np.sqrt(n_features)))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("sklearn RF Accuracy:", accuracy)

    # Plot().plot_in_2d(X_test, y_pred, title="Random Forest", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()