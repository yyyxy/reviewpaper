import csv


fr = open('../data/query_final_sta.csv', 'r', encoding='utf-8')
reader = csv.reader(fr)
query = []
answer = []
idx = []
line = 0
for row in reader:
    if int(row[2]) < 11 and int(row[2]) > 0:
        idx.append(line)
        query.append(row[3])
        tmp = []
        print(row[4])
        for ap in row[4][1:-1].split(', '):
            print(ap[1:-1])
            tmp.append(ap[1:-1])
        answer.append(tmp)
    line += 1

print(len(query), len(idx))
print(idx)


# 根据筛选的数据，得到其对应的get_rec和get_feature数据
def get_rec_feature_data(idx):
    fr = open('../data/get_rec_biker.csv', 'r', encoding='utf-8')
    reader = csv.reader(fr)
    data = []
    line = 0
    for row in reader:
        if int(line/10) in idx:
            data.append(row)
        line += 1

    fw = open('../data/get_rec_biker_filter.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(fw)
    for i in range(len(data)):
        writer.writerow(data[i])
    fw.close()


    fr = open('../data/get_feature_biker.csv', 'r', encoding='utf-8')
    reader = csv.reader(fr)
    data = []
    line = 0
    for row in reader:
        if int(line/10) in idx:
            data.append(row)
        line += 1

    fw = open('../data/get_feature_biker_filter.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(fw)
    for i in range(len(data)):
        writer.writerow(data[i])
    fw.close()


# 从query_final_sta文件中筛选出ground-truth在1-10之间的数据
def get_filter_data():
    fw = open('../data/feedback_all_original_biker_filter.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(fw)
    for i in range(len(query)):
        writer.writerow([query[i]]+answer[i])
    fw.close()


get_rec_feature_data(idx)

