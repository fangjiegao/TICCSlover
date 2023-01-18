# coding=utf-8
from sklearn.covariance import GraphicalLassoCV
# import numpy as np


def get_GraphicalLassoCV(X):
    X -= X.mean(axis=0)
    X /= X.std(axis=0)  # Z标准化:数据均值为0，方差为1.
    model = GraphicalLassoCV()
    model.fit(X)
    """
    cov_ = model.covariance_  # lasso协方差
    prec_ = model.precision_  # 协方差的逆
    """
    return model.precision_, model.covariance_


def get_GraphicalLassoCV_for_multiprocessing(X, precision_list, covariance_dict, index, number_of_clusters):
    # X -= X.mean(axis=1, keepdims=True)  # 行归一化，部分数据可用
    X -= X.mean(axis=0)
    X += 0.0001
    x_std = X.std(axis=0)
    # x_std = np.maximum(x_std, 0.0001)
    x_std[x_std == 0] = 1
    X /= x_std  # Z标准化:数据均值为0，方差为1.
    model = GraphicalLassoCV(n_jobs=1, tol=1e-2)
    model.fit(X)
    """
    cov_ = model.covariance_  # lasso协方差
    prec_ = model.precision_  # 协方差的逆
    """
    precision_list[index] = model.precision_
    # print(number_of_clusters, index, model.covariance_)
    covariance_dict[number_of_clusters, index] = model.covariance_
