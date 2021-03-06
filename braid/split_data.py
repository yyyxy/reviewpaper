import csv
import random
from os import path


# path1 = 'I:/first_review_biker/first_review_data/'
path1 = 'D:/first_review_data/'

# split the '' element in each row of sim_test.csv
# def split_test():
#     fr = open(path.join(path1, './sim_test_class_2.csv'), 'r', encoding='utf-8')
#     reader = csv.reader(fr)
#     queries, test_query, test_answer, train_query, train_answer, index_test, test_row = [], [], [], [], [], [], []
#     fw = open(path.join(path1, './sim_test_3.csv'), 'w', newline='', encoding='utf-8')
#     writer = csv.writer(fw)
#     for row in reader:
#         temp = []
#         for i in row:
#             if i != '':
#                 temp.append(i)
#         print(temp)
#         test_row.append(temp)
#
#     fr = open(path.join(path1, './feedback_all_class.csv'), 'r')
#     reader = csv.reader(fr)
#     for row in reader:
#         queries.append(row[0])
#         if row in test_row:
#             index_test.append(queries.index(row[0]))
#             test_query.append(row[0])
#             writer.writerow([row[0]]+row[1:])
#             print(row[0])
#             temp = []
#             for i in range(len(row[1:])):
#                 if row[i+1] != '':
#                     temp.append(row[i+1].lower())
#             test_answer.append(temp)
#         else:
#             train_query.append(row[0])
#             train_answer.append(row[1:])


# split the testing and training set
# def get_test_train():
#     queries, test_query, test_answer, train_query, train_answer, index_test, test_row = [], [], [], [], [], [], []
#     fr = open(path.join(path1, './sim_test_3.csv'), 'r', encoding='utf-8')
#     reader = csv.reader(fr)
#     for row in reader:
#         test_row.append(row)
#         print(row[0])
#     print(test_row)
#
#     fr = open(path.join(path1, './feedback_all_class.csv'), 'r')
#     reader = csv.reader(fr)
#     for row in reader:
#         queries.append(row[0])
#         if row in test_row:
#             index_test.append(queries.index(row[0]))
#             test_query.append(row[0])
#             print(row[0])
#             temp = []
#             for i in range(len(row[1:])):
#                 if row[i+1] != '':
#                     temp.append(row[i+1].lower())
#             test_answer.append(temp)
#         else:
#             train_query.append(row[0])
#             train_answer.append(row[1:])
#
#     fr = open(path.join(path1, './get_feature_class.csv'), 'r')
#     reader = csv.reader(fr)
#     feat, rec_api_test, rec_api_train, test_feature, train_feature = [], [], [], [], []
#     for row in reader:
#         feat.append(row)
#
#     for i in range(len(feat)):
#         if int(i/10) in index_test:
#             test_feature.append(feat[i][1:-1])
#             rec_api_test.append(feat[i][-1])
#         else:
#             train_feature.append(feat[i][:-1])
#             rec_api_train.append(feat[i][-1])
#     print(111, len(test_query), len(train_query))
#     return test_query, test_answer, train_query, train_answer, test_feature, train_feature, rec_api_test, rec_api_train


def idx_to_data(idx):
    # 索引转化为数据
    fr = open(path.join(path1, './feedback_all.csv'), 'r')
    reader = csv.reader(fr)
    idx_query, idx_answer = [], []
    for i, row in enumerate(reader):
        if i in idx:
            idx_query.append(row[0])
            idx_answer.append(row[1:])

    fr = open(path.join(path1, './get_feature_method_fdbtag.csv'), 'r')
    reader = csv.reader(fr)
    idx_rec_api, idx_feature = [], []
    for i, row in enumerate(reader):
        if int(i/10) in idx:
            idx_feature.append(row[:-1])
            idx_rec_api.append(row[-1])

    return idx_query, idx_answer, idx_rec_api, idx_feature


def split_choose_unlabel(train_query, train_answer, rec_api_train, num):
    count = list(range(len(train_query)))
    choose = random.sample(count, num)
    print('choose', choose, len(choose))

    choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel = [], [], [], [], [], []
    for i in range(len(train_query)):
        if i in choose:
            choose_query.append(train_query[i])
            choose_answer.append(train_answer[i])
            for n in range(10):
                rec_api_choose.append(rec_api_train[10*i+n])
        else:
            unlabel_query.append(train_query[i])
            unlabel_answer.append(train_answer[i])
            for n in range(10):
                rec_api_unlabel.append(rec_api_train[10*i+n])
    print(111,len(choose_query), len(unlabel_query))
    return choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose


def get_add_FR_rec_api(unlabel_query, rec_api_unlabel, add_choose):
    rec_api_choose = []
    for i in range(len(unlabel_query)):
        if i in add_choose:
            for n in range(10):
                rec_api_choose.append(rec_api_unlabel[10*i+n])
    return rec_api_choose


def split_10_choose_unlabel(train_query, train_answer, rec_api_train):
    choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose = [], [], [], [], [], [], []
    # file = open('../data/10_query_1.txt', 'r')
    # for f in file.readlines():
    #     # print(f.split('\n')[0])
    #     choose_query.append(f.split('\n')[0])

    fr = open(path.join(path1, './feedback_random.csv'), 'r')
    reader = csv.reader(fr)
    for row in reader:
        choose_query.append(row[0])
        choose_answer.append(row[1:])

    for i in range(len(train_query)):
        if train_query[i] in choose_query:
            choose.append(i)
            # choose_answer.append(train_answer[i])
            # for n in range(10):
            #     rec_api_choose.append(rec_api_train[10*i+n])
        else:
            unlabel_query.append(train_query[i])
            unlabel_answer.append(train_answer[i])
            for n in range(10):
                rec_api_unlabel.append(rec_api_train[10*i+n])
    return choose_query, choose_answer, unlabel_query, unlabel_answer, rec_api_unlabel, choose


def get_choose_10_query(train_feature, choose):
    choose_feature, unlabel_feature, rec_api_choose = [], [], []

    fr = open(path.join(path1, './feedback_feature_random.csv'), 'r')
    reader = csv.reader(fr)
    for row in reader:
        choose_feature.append(row[:-1])
        rec_api_choose.append(row[-1])

    for i in range(len(train_feature)):
        if int(i/10) in choose:
            choose_feature.append(train_feature[i])
        else:
            unlabel_feature.append(train_feature[i])

    return choose_feature, unlabel_feature, rec_api_choose


def get_choose(train_feature, choose):
    choose_feature, unlabel_feature = [], []

    for i in range(len(train_feature)):
        if int(i/10) in choose:
            choose_feature.append(train_feature[i])
        else:
            unlabel_feature.append(train_feature[i])

    return choose_feature, unlabel_feature


def get_train_feature_matrix(feedback_inf, choose_feature):
    X, y, line = [], [], 0

    for row in choose_feature:
        x = []
        yy = float(row[0])
        x.append(float(row[1]))
        x.append(float(row[2]))
        x.extend(feedback_inf[line])
        X.append(x)
        y.append(int(yy))
        line += 1
    return X, y


def get_test_feature_matrix(feedback_inf, test_feature):
    X = []
    line = 0

    for row in test_feature:
        x = []
        x.append(float(row[1]))
        x.append(float(row[2]))
        x.extend(feedback_inf[line])
        X.append(x)
        line += 1
    # print('len(feedback_inf)', len(feedback_inf))
    # print(len(X))
    return X


if __name__ == "__main__":
	split_test()
	index_test, test_query, train_query = get_test_train()
	test_feature, train_feature, choose = get_feature(index_test, train_query, 13)
	get_AL_feature(train_feature, choose)
