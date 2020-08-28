import csv
import math
import split_data

# split_data.split_test()

# test preprocessing
fr = open('../data/feedback_all_new_nlp.csv', 'r')
reader = csv.reader(fr)
test_query = []
test_answer = []
for row in reader:
    test_query.append(row[0])
    temp_api = [n.lower() for n in row[1:] if len(n) > 1]
    test_answer.append(temp_api)
print(test_query)
print(test_answer)

fr = open('../data/get_feature_new_nlp.csv', 'r')
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
            print(test_answer[int(count/10)])
            print(temp_api)
    count += 1


sort, sort_all = [], []
for i in range(len(test_query)):
    tmp_all = []
    for api in test_answer[i]:
        tmp_rec_api = rec_api[i]
        if api in tmp_rec_api and len(api) > 1:
            tmp_all.append(tmp_rec_api.index(api) + 1)
    tmp_all = sorted(tmp_all)
    sort_all.append(tmp_all)
    if len(tmp_all) == 0:
        sort.append(-1)
    else:
        sort.append(tmp_all[0])
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
    if i <= 3 and i>-1:
        top3 += 1
    if i <= 5 and i>-1:
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

fw = open('../data/miss_nlp.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(fw)
for i in range(len(sort_all)):
    if len(sort_all[i]) == 0:
        writer.writerow([test_query[i]]+test_answer[i])
