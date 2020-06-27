import numpy as np
from decision_tree import DecisionTree, ClassificationTree

class RandomForest():
    """Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.
    Parameters:
    -----------
    n_estimators: int
        树的数量
        The number of classification trees that are used.
    max_features: int
        每棵树选用数据集中的最大的特征数
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        每棵树中最小的分割数，比如 min_samples_split = 2表示树切到还剩下两个数据集时就停止
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        每棵树切到小于min_gain后停止
        The minimum impurity required to split the tree further.
    max_depth: int
        每棵树的最大层数
        The maximum depth of a tree.
    """

    def __init__(self, n_estimators=100, min_samples_split=2, min_gain=0,
                 max_depth=float("inf"), max_features=None):

        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.max_features = max_features

        self.trees = []
        # 建立森林(bulid forest)
        for _ in range(self.n_estimators):
            tree = ClassificationTree(min_samples_split=self.min_samples_split, min_impurity=self.min_gain,
                                      max_depth=self.max_depth, max_features = max_features)
            self.trees.append(tree)

    def fit(self, X, y):
        sub_sets = self.get_bootstrap_data(X, y)
        n_features = X.shape[1]
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            sub_X, sub_y = sub_sets[i]
            self.trees[i].fit(sub_X, sub_y)

    # 有放回采样
    def get_bootstrap_data(self, X, y):
        m = X.shape[0]
        y = y.reshape(m, 1)

        Xy = np.hstack((X, y))
        np.random.shuffle(Xy)

        data_sets = []
        for _ in range(self.n_estimators):
            idm = np.random.choice(m, m, replace=True) # 从m个数中有放回的选m个数
            bootstrap_X = Xy[idm, :-1]
            bootstrap_y = Xy[idm, -1:]
            data_sets.append([bootstrap_X, bootstrap_y])
        return data_sets

    def predict(self, X):
        y_preds = []
        for i in range(self.n_estimators):
            y_pred = self.trees[i].predict(X)
            y_preds.append(y_pred)
        y_preds = np.array(y_preds).T
        y_pred = []
        for y_p in y_preds:
            # np.bincount()可以统计每个索引出现的次数
            # np.argmax()可以返回数组中最大值的索引
            # cheak np.bincount() and np.argmax() in numpy Docs
            y_pred.append(np.bincount(y_p.astype('int')).argmax())
        return y_pred