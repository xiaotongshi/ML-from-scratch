import matplotlib.pyplot as plt

def roc_curve(y_true, y_score):
    db = list(zip(y_true, y_score))
    rank = [label for label, score in sorted(db, key=lambda x: x[1], reverse=True)] #按照score降序排列
    real_P = sum(y_true) # real label is positive
    real_N = len(y_true) - real_P # real label is negative

    #计算ROC坐标点
    xy_arr = []
    tp, fp = 0, 0			
    for i in range(len(rank)):
        if rank[i] == 1:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/real_N,tp/real_P])

    #计算曲线下面积
    auc = 0			
    prev_x = 0
    for x,y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x

    # 画图
    x = [_v[0] for _v in xy_arr]
    y = [_v[1] for _v in xy_arr]
    plt.title("ROC curve of %s (AUC = %.4f)" % ('svm',auc))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(x, y)# use pylab to plot x and y
    # plt.show()# show the plot on the screen

    return auc

def roc_auc_score(y_true, y_score, method=1):
    db = list(zip(y_true, y_score))
    rank = [label for label, score in sorted(db, key=lambda x: x[1], reverse=True)] #按照score降序排列
    real_P = sum(y_true) # real label is positive
    real_N = len(y_true) - real_P # real label is negative

    if method == 1:
        # 计算loss
        p = []
        n = []
        loss = 0
        for i in range(len(y_score)):
            if y_true[i] == 1:
                p.append(y_score[i])
            else:
                n.append(y_score[i])
        for i in range(len(p)):
            for j in range(len(n)):
                if n[j] > p[i]:
                    loss += 1
                elif n[j] == p[i]:
                    loss += 0.5
        auc = 1 - loss/(len(p)*len(n))
        # print("the auc is %s."%auc)

    elif method == 2:
        # 用排序误差计算
        accumulated_neg = 0
        rank_loss = 0
        for i in range(len(rank)):
            if rank[i] == 1:
                rank_loss += accumulated_neg # 目前有多少个负样本分数大于正样本
            else:
                accumulated_neg += 1
        rank_loss /= (real_P*real_N)
        auc = 1 - rank_loss
        # print("the auc is %s."%auc)

    elif method == 3:
        # 概率从小到大排序，排在第rank的位置。最小的排名1
        n = len(rank)
        rank_pos = [n-i for i, label in enumerate(rank) if label == 1]
        auc = (sum(rank_pos) - real_P*(real_P+1)/2) / (real_P * real_N)

    else:
        print('未定义method')
    return auc

def main():
    # preds = [0.9, 0.4, 0.3, 0.2, 0.6, 0.5]
    # labels = [1, 1, 0, 0, 1, 0]
    y_score = [0.9, 0.7, 0.6, 0.55, 0.52, 0.4, 0.38, 0.35, 0.31, 0.1]
    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    auc = roc_curve(y_true, y_score)
    print("the auc is %s."%auc)
    auc = roc_auc_score(y_true, y_score)
    print("the auc is %s."%auc)
    auc = roc_auc_score(y_true, y_score, method=2)
    print("the auc is %s."%auc)
    auc = roc_auc_score(y_true, y_score, method=3)
    print("the auc is %s."%auc)

if __name__ == "__main__":
    main()