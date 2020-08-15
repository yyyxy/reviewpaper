import csv
import random
import feedback
from preprocess import similarity


# # split the '' element in each row of sim_test.csv
# def split_test():
#     fr = open('../data/sim_test_final_2_new.csv', 'r', encoding='utf-8')
#     reader = csv.reader(fr)
#     fw = open('../data/sim_test_temp.csv', 'w', newline='')
#     writer = csv.writer(fw)
#     for row in reader:
#         temp = []
#         for i in row:
#             if i != '':
#                 temp.append(i)
#         print(temp)
#         writer.writerow(temp)


def split_10_choose_unlabel(train_query, train_answer, rec_api_train):
    choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose = [], [], [], [], [], [], []
    file = open('../data/10_query_5.txt', 'r')
    for f in file.readlines():
        # print(f.split('\n')[0])
        choose_query.append(f.split('\n')[0])

    for i in range(len(train_query)):
        if train_query[i] in choose_query:
            choose.append(i)
            choose_answer.append(train_answer[i])
            for n in range(10):
                rec_api_choose.append(rec_api_train[10*i+n])
        else:
            unlabel_query.append(train_query[i])
            unlabel_answer.append(train_answer[i])
            for n in range(10):
                rec_api_unlabel.append(rec_api_train[10*i+n])
    return choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose


# # split the testing and training set
# def get_test_train():
#     queries, test_query, test_answer, train_query, train_answer, index_test, test_row = [], [], [], [], [], [], []
#     fr = open('../data/sim_test_temp.csv', 'r')
#     reader = csv.reader(fr)
#     for row in reader:
#         test_row.append(row)
#
#     fr = open('../data/feedback_all.csv', 'r')
#     reader = csv.reader(fr)
#     for row in reader:
#         queries.append(row[0])
#         if row in test_row:
#             index_test.append(queries.index(row[0]))
#             test_query.append(row[0])
#             temp = []
#             for i in range(len(row[1:])):
#                 if row[i+1] != '':
#                     temp.append(row[i+1].lower())
#             test_answer.append(temp)
#         else:
#             train_query.append(row[0])
#             train_answer.append(row[1:])
#
#     fr = open('../data/get_feature.csv', 'r')
#     reader = csv.reader(fr)
#     feat, rec_api_test, test_feature, train_feature, rec_api_train = [], [], [], [], []
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
#
#     return test_query, test_answer, train_query, train_answer, test_feature, train_feature, rec_api_test, rec_api_train


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
    return choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose


def get_add_FR_rec_api(unlabel_query, rec_api_unlabel, add_choose):
    rec_api_choose = []
    for i in range(len(unlabel_query)):
        if i in add_choose:
            for n in range(10):
                rec_api_choose.append(rec_api_unlabel[10*i+n])
    return rec_api_choose


def get_choose(train_feature, choose):
    choose_feature, unlabel_feature = [], []

    for i in range(len(train_feature)):
        if int(i/10) in choose:
            choose_feature.append(train_feature[i])
        else:
            unlabel_feature.append(train_feature[i])

    return choose_feature, unlabel_feature


def idx_to_data(idx):
    # 索引转化为数据
    fr = open('../data/feedback_all_new_rack.csv', 'r')
    reader = csv.reader(fr)
    idx_query, idx_answer = [], []
    for i, row in enumerate(reader):
        if i in idx:
            idx_query.append(row[0])
            idx_answer.append(row[1:])

    fr = open('../data/get_feature_new_rack.csv', 'r')
    reader = csv.reader(fr)
    idx_rec_api, idx_feature = [], []
    for i, row in enumerate(reader):
        if int(i/10) in idx:
            idx_feature.append(row[:-1])
            idx_rec_api.append(row[-1])

    return idx_query, idx_answer, idx_rec_api, idx_feature


def get_choose_data(choose_idx, test_query, pct, w2v, idf):
    matrix, idf_vector = [], []
    for query in test_query:
        query_matrix, query_idf_vector = feedback.load_matrix(query, w2v, idf)
        matrix.append(query_matrix)
        idf_vector.append(query_idf_vector)

    # 索引转化为数据
    choose_query, choose_answer, choose_rec_api, choose_feature = idx_to_data(choose_idx)
    print(len(choose_query), len(choose_rec_api))
    idx = []
    for i in range(len(choose_query)):
        q1_matrix, q1_idf_vector = feedback.load_matrix(choose_query[i], w2v, idf)
        for n in range(len(matrix)):
            q_sim = similarity.sim_doc_pair(q1_matrix, matrix[n], q1_idf_vector, idf_vector[n])
            # if q_sim > 0.65-round(pct, 2)*0.1:
            if q_sim > 0.65:
                # print(i, q_sim)
                idx.append(i)
                break

    query, answer, rec_api, feature = [], [], [], []
    for i in idx:
        query.append(choose_query[i])
        answer.append(choose_answer[i])
        for n in range(10):
            feature.append(choose_feature[i*10+n])
            rec_api.append(choose_rec_api[i*10+n])
    print('len(choose_feature)', len(query), len(feature))

    return query, answer, rec_api, feature


def get_unlabel_data(test_query, w2v, idf):
    matrix, idf_vector = [], []
    for query in test_query:
        query_matrix, query_idf_vector = feedback.load_matrix(query, w2v, idf)
        matrix.append(query_matrix)
        idf_vector.append(query_idf_vector)

    # so相关数据
    # 索引转化为数据
    fr = open('../data/feedback_repository_rack_oracle.csv', 'r')
    reader = csv.reader(fr)
    idx = []
    query, answer = [], []
    for i, row in enumerate(reader):
        q1_matrix, q1_idf_vector = feedback.load_matrix(row[0], w2v, idf)
        for n in range(len(matrix)):
            q_sim = similarity.sim_doc_pair(q1_matrix, matrix[n], q1_idf_vector, idf_vector[n])
            if q_sim > 0.6 and q_sim < 1:
                # if q_sim > 0.8:
                # print(1111111, row[0])
                query.append(row[0])
                answer.append(row[1:])
                idx.append(i)
                # print(i, q_sim)
                break
    print('ground_truth_training', len(idx), idx)

    fr = open('../data/get_feature_rack_oracle.csv', 'r')
    reader = csv.reader(fr)
    rec_api, feature = [], []
    for i, row in enumerate(reader):
        if int(i/10) in idx:
            feature.append(row[:-1])
            rec_api.append(row[-1])

    # training data除反馈数据之外的其他数据

    return query, answer, rec_api, feature


def get_train_feature_matrix(feedback_inf, choose_feature):
    X, y, line = [], [], 0

    for row in choose_feature:
        x = []
        yy = float(row[0])
        x.append(float(row[1]))
        # x.append(float(row[2]))
        x.extend(feedback_inf[line])
        # for fr in feedback_inf[line]:
        #     if float(row[1]) != 0.0:
        #         x.append(fr/float(row[1]))
        #     else:
        #         x.append(0)
        #     if float(row[2]) != 0.0:
        #         x.append(fr/float(row[2]))
        #     else:
        #         x.append(0)
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
        # x.append(float(row[2]))
        x.extend(feedback_inf[line])
        # for fr in feedback_inf[line]:
        #     if float(row[1]) != 0.0:
        #         x.append(fr/float(row[1]))
        #     else:
        #         x.append(0)
        #     if float(row[2]) != 0.0:
        #         x.append(fr/float(row[2]))
        #     else:
        #         x.append(0)
        X.append(x)
        line += 1
    print('len(feedback_inf)', len(feedback_inf))
    print(len(X))
    return X
