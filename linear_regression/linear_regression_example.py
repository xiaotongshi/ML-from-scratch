import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

from linear_regression import LinearRegression
from linear_regression import l1_regularization
from linear_regression import l2_regularization


def main():
    X, y = datasets.make_regression(n_samples=500, n_features=1, noise=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 比较sklearn上的模型和自己写的模型
    lr1 = linear_model.LinearRegression()
    lr2 = LinearRegression()
    lasso1 = linear_model.Lasso(alpha=0.1)
    lasso2 = LinearRegression(l1=0.1)
    ridge1 = linear_model.Ridge(alpha=0.5)
    ridge2 = LinearRegression(l2=0.5)
    elasticnet1 = linear_model.ElasticNet(alpha=0.5, l1_ratio=0.5)
    elasticnet2 = LinearRegression(l1=0.25, l2=0.25*0.5)
    

    models = {'linear1': lr1, 'linear': lr2, 'lasso1': lasso1, 'lasso2': lasso2, 
            'ridge1': ridge1, 'ridge2': ridge2, 'elasticnet1': elasticnet1, 'elasticnet2': elasticnet2}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.reshape(y_pred, y_test.shape)
        mse = mean_squared_error(y_test, y_pred)
        print('{}: {}'.format(model_name, mse))


if __name__ == "__main__":
    main()