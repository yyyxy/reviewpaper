import csv

query = []
answer = []
fr = open('../data/nlp_filter_so_groundtruth_comple2.csv', 'r', encoding='utf-8')
reader = csv.reader(fr)
for row in reader:
    query.append(row[0])
    print(type(row[1]), row[1][1:-1])
    tmp = []
    for i in row[1][1:-1].split(", "):
        api_class = i[1:-1]
        if api_class not in tmp:
            tmp.append(api_class)
        print(api_class, tmp)
    answer.append(tmp)

    # tmp = []
    # count = 0
    # for i in row[1].split("'"):
    #     if count%2 == 1:
    #         api_class = i.split('.')[-2]
    #         if api_class not in tmp:
    #             tmp.append(api_class)
    #         print(api_class, tmp)
    #     count += 1
    # answer.append(tmp)

fw = open('../data/nlp_filter_so_groundtruth_comple3.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(fw)
for i in range(len(query)):
    writer.writerow([query[i]]+answer[i])
fw.close()

