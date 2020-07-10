from sklearn.model_selection import KFold
import pandas as pd
import csv


# fr = open('D:/first_review_data/get_rec_class.csv', 'r')
# reader = csv.reader(fr)
# for row in reader:
#     print(row[0], row[1])

# train = pd.DataFrame([[1,2,3,4,5,6],[3,3,3,3,3,3],[4,4,4,4,4,4],[5,5,5,5,5,5],[6,6,6,6,6,6],[7,7,7,7,7,7]])#训练集
# test = pd.DataFrame([0,0,0,1,1,1])#测试集
# kf = KFold(n_splits = 6,random_state = 2,shuffle = True)#实例化，配置三个参数
# for i, j in kf.split(train, test):#设置6折，便会循环6次
#     print(i, j)

import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
for train_idx, test_idx in kf.split(X):
    print("%s %s" % (train_idx, test_idx))
    # print(X[int(test_idx)])


