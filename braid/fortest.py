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

# import numpy as np
# from sklearn.model_selection import KFold
#
# X = ["a", "b", "c", "d"]
# kf = KFold(n_splits=2)
# for train_idx, test_idx in kf.split(X):
#     print("%s %s" % (train_idx, test_idx))
#     # print(X[int(test_idx)])

# fr = open('D:/first_review_data/get_feature_method.csv', 'r')
fr = open('D:/first_review/rack/data/get_feature_nlp.csv', 'r')
reader = csv.reader(fr)
feature = []
for row in reader:
    feature.append(row)

# fw = open('D:/first_review_data/get_feature_method_fdbtag2.csv', 'w', newline='')
fw = open('D:/first_review/rack/data/get_feature_nlp_fdbtag2.csv', 'w', newline='')
writer = csv.writer(fw)
for row in feature:
    label = 0
    if int(row[0]) > 5:
        label = 1
        # label = round((int(row[0])-5)/5, 1)
    # for i in range(1, 3):
    #     if float(row[i]) > 0:
    #         row[i] = round(float(row[i]), 2)
    for i in range(3, 8):
        if float(row[i]) > 0:
            row[i] = round((float(row[i])-0.6)/0.4, 2)
    writer.writerow([row[0]]+row[1:3]+[label]+row[3:-1]+[row[-1]])
