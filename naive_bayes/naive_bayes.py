import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

class NaiveBayes():
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.parameters = {}
        for c in self.classes:
            xc = X[np.where(y == c)]
            Pxc_mean = np.mean(xc, axis=0, keepdims=True)
            Pxc_var = np.var(xc, axis=0, keepdims=True)
            parameters = {'mean':Pxc_mean, 'var': Pxc_var, 'prior': xc.shape[0]/X.shape[0]}
            self.parameters['class' + str(c)] = parameters

    def _pdf(self, X, classes):
        # 一维高斯分布的概率密度函数P(x|c)
        # eps为防止分母为0
        eps = 1e-4
        mean = self.parameters["class" + str(classes)]["mean"]
        var = self.parameters["class" + str(classes)]["var"]

        numerator = np.exp(-(X - mean) ** 2 / (2 * var + eps))
        denominator = np.sqrt(2 * np.pi * var + eps)

        # 取对数防止数值溢出
        result = np.sum(np.log(numerator / denominator), axis=1, keepdims=True)
        return result.T 

    def _predict(self, X):
        # 计算每个种类的概率P(Y|x1,x2,x3) =  P(Y)*P(x1|Y)*P(x2|Y)*P(x3|Y)
        output = []
        for y in range(self.classes.shape[0]):
            prior = np.log(self.parameters['class'+str(y)]['prior'])
            posterior = self._pdf(X, y)
            prediction = prior + posterior # log(prior) + log(posterior) = log(prior * posterior)
            output.append(prediction)
        return output

    def predict(self, X):
        # 取概率最大的类别返回预测值
        output = self._predict(X)
        output = np.reshape(output, (self.classes.shape[0], X.shape[0]))
        prediction = np.argmax(output, axis=0)
        return prediction

def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print("X_train",X_train.shape)
    clf = NaiveBayes()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # # Reduce dimension to two using PCA and plot the results
    # Plot().plot_in_2d(X_test, y_pred, title="Naive Bayes", accuracy=accuracy, legend_labels=data.target_names)

if __name__ == "__main__":
    main()