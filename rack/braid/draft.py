import csv


def get_gt_file():
    fr = open('../data/example_original.csv', 'r')
    reader = csv.reader(fr)
    query = []
    answer = []
    ori_pos = []
    for row in reader:
        ori_pos.append(row[0])
        query.append(row[1])
        ans = row[2][1:-1]
        tmp = []
        for i in ans.split(', '):
            print(i[1:-1].split('.')[-2])
            tmp_class = i[1:-1].split('.')[-2]
            if tmp_class not in tmp:
                tmp.append(tmp_class)
        print(tmp)
        answer.append(tmp)

    fw = open('../data/example_nlp.csv', 'w', newline='', encoding ='utf-8')
    writer = csv.writer(fw)
    for i in range(len(query)):
        writer.writerow([query[i]]+answer[i])


def get_rank_state():
    fr = open('../data/example_biker.csv', 'r')
    reader = csv.reader(fr)
    test_query = []
    test_answer = []
    for row in reader:
        test_query.append(row[0])
        temp_api = [n.lower() for n in row[1:] if len(n) > 1]
        test_answer.append(temp_api)
    print(len(test_query), test_query)
    print(len(test_answer), test_answer)

    fr = open('../data/get_feature_biker_example.csv', 'r')
    reader = csv.reader(fr)

    queries = []
    rec_api, temp_api = [], []
    count = 0
    for row in reader:
        if count % 30 == 0:
            queries.append(row[0])
            temp_api = [row[-1].lower()]
        else:
            temp_api.append(row[-1].lower())
            if count % 30 == 29:
                rec_api.append(temp_api)
                print(test_answer[int(count/30)])
                print(temp_api)
        count += 1
    print(count)
    print(len(queries), len(rec_api))

    sort, sort_all = [], []
    for i in range(len(test_query)):
        tmp_all = []
        for api in test_answer[i]:
            tmp_rec_api = rec_api[i*30:i*30+30]
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


get_rank_state()

