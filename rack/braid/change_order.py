import csv
from os import path


# 第一步，给feedback数据标上索引
def index_feedback_data():
    fr = open('../data/feedback_all_original_biker.csv', 'r')
    reader = csv.reader(fr)
    row = []
    for r in reader:
        row.append(r)

    fw = open('../data/feedback_all_new_biker_idx.csv', 'w', newline='')
    writer = csv.writer(fw)
    for i in range(len(row)):
        writer.writerow([i+1]+row[i])


# 第二步，根据索引，更改get_rec和get_feature的顺序
def get_new_order():
    # fr = open('../data/feedback_all_new_rack.csv', 'r')
    # reader = csv.reader(fr)
    # idx = []
    # for r in reader:
    #     idx.append(int(r[0]))
    # print(idx)

    idx = del_index()

    # get_rec
    fr = open('../data/get_rec_biker.csv', 'r')
    reader = csv.reader(fr)
    rec = []
    for r in reader:
        rec.append(r)

    fw = open('../data/get_rec_new_biker.csv', 'w', newline='')
    writer = csv.writer(fw)
    for i in idx:
        for n in range(10):
            writer.writerow(rec[(i-1)*10+n])

    # get_feature
    fr = open('../data/get_feature_biker.csv', 'r')
    reader = csv.reader(fr)
    rec = []
    for r in reader:
        rec.append(r)

    fw = open('../data/get_feature_new_biker.csv', 'w', newline='')
    writer = csv.writer(fw)
    for i in idx:
        for n in range(10):
            writer.writerow(rec[(i-1)*10+n])


def get_index():
    fr = open('../data/feedback_all_original_biker.csv', 'r')
    reader = csv.reader(fr)
    row = []
    for r in reader:
        row.append(r[0])

    fr = open('../data/feedback_all_new_biker.csv', 'r')
    reader = csv.reader(fr)
    idx = []
    for r in reader:
        for q in row:
            if r[0] == q:
                idx.append(row.index(q)+1)
    print(idx)

    for i in range(1, 414):
        if i not in idx:
            print(i)

    return idx


def del_index():
    fr = open('../data/feedback_all_new_biker_idx.csv', 'r')
    reader = csv.reader(fr)
    row = []
    idx = []
    for r in reader:
        idx.append(int(r[0]))
        row.append(r[1:])
    print(idx)

    fw = open('../data/feedback_all_new_biker.csv', 'w', newline='')
    writer = csv.writer(fw)
    for i in row:
        writer.writerow(i)
    return idx

# index_feedback_data()
get_new_order()
del_index()

