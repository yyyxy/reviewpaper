import csv
from os import path
import split_data

# split_data.split_test()
path1 = 'D:/first_review_data/'

# test preprocessing
fr = open(path.join(path1, './feedback_all.csv'), 'r')
reader = csv.reader(fr)
test_query = []
test_answer = []
for row in reader:
    test_query.append(row[0])
    temp_api = [n.lower() for n in row[1:]]
    test_answer.append(temp_api)
print(test_query)
print(test_answer)

fr = open(path.join(path1, './get_rec_method.csv'), 'r')
reader = csv.reader(fr)

queries = []
rec_api, temp_api = [], []
count = 0
for row in reader:
    if count%10 == 0:
        queries.append(row[0])
        temp_api = [row[1].lower()]
    else:
        temp_api.append(row[1].lower())
        if count%10 == 9:
            rec_api.append(temp_api)
    count += 1


sort, sort_all = [], []
for i in range(len(test_query)):
    flag = 0
    if test_query[i] in queries:
        index = 1
        temp_all = []
        for api in rec_api[queries.index(test_query[i])]:
            if api.lower() in test_answer[i]:
                # print(i, index, temp_all)
                temp_all.append(index)
                if len(temp_all) > 0 and flag == 0:
                    sort.append(temp_all[0])
                    flag = 1
            index += 1
        sort_all.append(temp_all)
print(sort)
print(sort_all)


top1, top3, top5, map, mrr, miss = 0, 0, 0, 0, 0, 0
for i in sort:
    if i == 1:
        top1 += 1
    if i <= 3:
        top3 += 1
    if i <= 5:
        top5 += 1
    mrr += 1/i

for n in sort_all:
    temp = 0
    count = 0
    for i in range(len(n)):
        count += 1
        temp += count / n[i]
    if len(n) != 0:
        temp = temp / len(n)
    if len(n) == 0:
        miss += 1
    map += temp

print(top1/len(test_query), top3/len(test_query), top5/len(test_query), map/len(test_query), mrr/len(test_query), miss)
# fw = open('../data/biker_metric_shareFR.csv', 'a+', newline='')
# writer = csv.writer(fw)
# writer.writerow(('original', top1/len(test_query), top3/len(test_query), top5/len(test_query), map/len(test_query), mrr/len(test_query)))
# fw.close()

