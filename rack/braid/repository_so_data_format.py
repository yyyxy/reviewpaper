import csv
import numpy as np


fr = open('../data/merge_nlp.csv', 'r')
reader = csv.reader(fr)

query = []
answer = []
for row in reader:
    print(row[1], row[2])
    query.append(row[1])
    answer.append(row[2])

fw = open('../data/feedback_repository_nlp_sim75.csv', 'w', newline='')
writer = csv.writer(fw)
for i, api in enumerate(answer):
    if i > 0:
        tmp = []
        for ap in api.split(','):
            tmp.append(ap)
        writer.writerow([query[i]] + tmp)


