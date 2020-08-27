import csv
import math
import split_data

# split_data.split_test()

# test preprocessing
fr = open('../data/feedback_all_new_biker.csv', 'r')
reader = csv.reader(fr)
test_query = []
test_answer = []
for row in reader:
    test_query.append(row[0])
    temp_api = [n.lower() for n in row[1:] if len(n) > 1]
    test_answer.append(temp_api)
print(test_query)
print(test_answer)

fr = open('../data/get_feature_new_biker.csv', 'r')
reader = csv.reader(fr)

queries = []
rec_api, temp_api = [], []
count = 0
for row in reader:
    if count%10 == 0:
        queries.append(row[0])
        temp_api = [row[-1].lower()]
    else:
        temp_api.append(row[-1].lower())
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
                print(i, index, temp_all)
                temp_all.append(index)
                if len(temp_all) > 0 and flag == 0:
                    sort.append(temp_all[0])
                    flag = 1
            index += 1
        sort_all.append(temp_all)
print(sort)
print(sort_all)


top1, top3, top5, map, mrr, ndcg = 0, 0, 0, 0, 0, 0
for n in sort_all:
    dcg, idcg = 0, 0
    if len(n) > 0:
        for i in range(10):
            rel_dcg = 0
            if (i + 1) in n:
                rel_dcg = 1
            dcg += rel_dcg / math.log((i + 2), 2)

            rel_idcg = 0
            if i < len(n):
                rel_idcg = 1
            idcg += rel_idcg / math.log((i + 2), 2)
        ndcg += dcg / idcg

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
    map += temp

print(top1/len(test_query), top3/len(test_query), top5/len(test_query), map/len(test_query), mrr/len(test_query), ndcg/len(test_query))
# fw = open('../data/metric_biker.csv', 'a+', newline='')
# writer = csv.writer(fw)
# writer.writerow(('original', top1/len(test_query), top3/len(test_query), top5/len(test_query), map/len(test_query), mrr/len(test_query)))
# fw.close()
