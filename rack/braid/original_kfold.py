import csv
from sklearn.model_selection import KFold
import random
import split_data
import math


fr = open('../data/feedback_all_new_nlp.csv', 'r')
reader = csv.reader(fr)
queries = []
for i, row in enumerate(reader):
    queries.append(row[0])

fr = open('../data/get_feature_new_nlp.csv', 'r')
reader = csv.reader(fr)
rec_api = []
for i, row in enumerate(reader):
    rec_api.append(row[0])

test_idx = [n for n in range(310)]
# k-fold cross validation
# kf = KFold(n_splits=10)
# round = 1
# for train_idx, test_idx in kf.split(queries):
    # train_idx = list(train_idx)
test_idx = list(test_idx)
print('test_idx', len(test_idx), test_idx)

# 获取测试数据
test_query, test_answer, test_rec_api, test_feature = split_data.idx_to_data(test_idx)
sort, sort_all = [], []
for i in range(len(test_query)):
    print(test_query[i])
    print(test_answer[i])
    print(test_rec_api[i*10:i*10+10])
    flag = 0
    tmp_all = []
    for api in test_answer[i]:
        tmp_rec_api = test_rec_api[i*10:i*10+10]
        if api in tmp_rec_api and len(api) > 1:
            tmp_all.append(tmp_rec_api.index(api) + 1)
    tmp_all = sorted(tmp_all)
    sort_all.append(tmp_all)
    if len(tmp_all) == 0:
        sort.append(-1)
    else:
        sort.append(tmp_all[0])
print(len(sort), sort)
print(sort_all)

top1, top3, top5, map, mrr, ndcg = 0, 0, 0, 0, 0, 0
for i in sort:
    if i == 1:
        top1 += 1
    if i <= 3 and i > 0:
        top3 += 1
    if i <= 5 and i > 0:
        top5 += 1
    if i != -1:
        mrr += 1/i

for n in sort_all:
    temp = 0
    count = 0
    for i in range(len(n)):
        count += 1
        temp += count / n[i]
    if len(n) != 0:
        temp = temp / len(n)
    map += temp

print(top1/len(test_query), top3/len(test_query), top5/len(test_query), map/len(test_query), mrr/len(test_query))
# fw = open('../data/metric_nlp.csv', 'a+', newline='')
# writer = csv.writer(fw)
# writer.writerow(('BRAID', round, top1/len(test_query), top3/len(test_query), top5/len(test_query), map/len(test_query), mrr/len(test_query)))
# fw.close()
#
# round += 1


