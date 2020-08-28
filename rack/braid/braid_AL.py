import split_data, feedback, metric
import gensim
import _pickle as pickle
import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.linear_model import LogisticRegression
import csv
import math


def get_AL_feature(t_answer, t_rec_api, feature):
    training_feature = []
    for i, train in enumerate(t_answer):
        # print('train',i, train)
        for index, ap in enumerate(t_rec_api[i*30:i*30+30]):
            if ap in train:
                temp = [1]
            else:
                temp = [0]
            temp.extend(feature[i*30+index][1:])
            training_feature.append(temp)
            # print(temp)
    return training_feature


def max_list(lt):
    temp, max_str = 0, 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    for i in range(lt.count(max_str)):
        lt.remove(max_str)
    return max_str


def get_active_data(pre_feedback_inf, pre_feature):
    X_pool = []
    x_feature, y_feature = split_data.get_train_feature_matrix(pre_feedback_inf, pre_feature)
    for row in x_feature:
        x = []
        for val in range(len(row)):
            x.append(float(row[val]))
        X_pool.append(x)
    X_pool = np.array(X_pool)
    y_pool = np.array(y_feature)
    return X_pool, y_pool


def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]


def get_AL_predict(pct, test_feature, choose_feature, unlabel_feature, test_query, choose_query, choose_answer, unlabel_query, unlabel_answer, rec_api_test, rec_api_choose, rec_api_unlabel, w2v, idf):
    unlabel_feedback_info = feedback.get_feedback_inf(unlabel_query, choose_query, choose_answer, rec_api_unlabel, w2v, idf)
    label_feedback_info = feedback.get_feedback_inf(choose_query, choose_query, choose_answer, rec_api_choose, w2v, idf)
    X_train, y_train = get_active_data(unlabel_feedback_info, unlabel_feature)
    X_feedback, y_feedback = get_active_data(label_feedback_info, choose_feature)

    predict = []
    print('len', len(X_feedback), len(unlabel_query))
    if len(X_feedback) > 0:
        # initializing the active learner
        learner = ActiveLearner(
            estimator=LogisticRegression(penalty='l1', solver='liblinear'),
            X_training=X_feedback, y_training=y_feedback
        )

        if len(unlabel_query) > 0:
            # pool-based sampling
            n_queries = int(pct*150)
            print('n_queries', n_queries, len(unlabel_query))
            for idx in range(n_queries):
                query_idx, query_instance = uncertainty_sampling(classifier=learner, X=X_train)
                # print('uncertain', query_idx, X_train[query_idx], y_train[query_idx])
                idx = int(query_idx/30)
                # print(idx, len(X_train))
                learner.teach(
                    X=X_train[query_idx].reshape(1, -1),
                    y=y_train[query_idx].reshape(1, )
                )
                # print(idx, len(unlabel_query))
                # print(unlabel_query[idx], unlabel_answer[idx], rec_api_unlabel[idx*10:idx*10+10], rec_api_unlabel[idx*10:idx*10+10])
                # add queried instance into FR
                choose_query.append(unlabel_query[idx])
                choose_answer.append(unlabel_answer[idx])
                rec_api_choose.extend(rec_api_unlabel[idx*30:idx*30+30])
                choose_feature.extend(unlabel_feature[idx*30:idx*30+30])

                # remove queried instance from pool
                for i in range(30):
                    X_train = np.delete(X_train, idx*30, axis=0)
                    y_train = np.delete(y_train, idx*30)
                del unlabel_query[idx]
                del unlabel_answer[idx]
                del rec_api_unlabel[idx*30:idx*30+30]
                del unlabel_feature[idx*30:idx*30+30]
                if len(X_train) == 0:
                    break
    else:
        choose_query = unlabel_query[:30]
        choose_answer = unlabel_answer[:30]
        rec_api_choose = rec_api_unlabel[:300]
        choose_feature = unlabel_feature[:300]

    add_label_feedback_info = feedback.get_feedback_inf(choose_query, choose_query, choose_answer, rec_api_choose, w2v, idf)
    new_X_feedback, new_y_feedback = get_active_data(add_label_feedback_info, choose_feature)
    learner = ActiveLearner(
        estimator=LogisticRegression(penalty='l1', solver='liblinear'),
        X_training=new_X_feedback, y_training=new_y_feedback
    )
    feedback_info = feedback.get_feedback_inf(test_query, choose_query, choose_answer, rec_api_test, w2v, idf)
    X = split_data.get_test_feature_matrix(feedback_info, test_feature)
    print('new_X_feedback', len(new_X_feedback), len(new_y_feedback))
    # # 扩展的标记集（feedback repository数据）特征向量
    # fw = open('../data/train.csv', 'w', newline='')
    # writer = csv.writer(fw)
    # for i in range(len(new_y_feedback)):
    #     writer.writerow((new_y_feedback[i], new_X_feedback[i][0], new_X_feedback[i][1], new_X_feedback[i][2], new_X_feedback[i][3], new_X_feedback[i][4], new_X_feedback[i][5], new_X_feedback[i][6], rec_api_choose[i]))
    #
    # # 测试集特征向量
    # fw = open('../data/test.csv', 'w', newline='')
    # writer = csv.writer(fw)
    # for i, x in enumerate(X):
    #     writer.writerow(x+[rec_api_test[i]])

    X_test = np.array(X)
    # 用反馈数据学习过后的模型来预测测试数据
    for query_idx in range(len(X)):
        y_pre = learner.predict_proba(X=X_test[query_idx].reshape(1, -1))
        predict.append(float(y_pre[0, 1]))
        # predict.append(math.log(float(y_pre[0, 1])+1))
        # predict.extend(y_pre.tolist())
    # print(predict)
    # print('new_choose', len(choose_query), len(choose_answer))

    return predict, X, new_X_feedback, new_y_feedback#, choose_query, choose_answer, rec_api_choose, choose_feature
    # return predict, choose_query, choose_answer, rec_api_choose, choose_feature


if __name__ == "__main__":
    w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed')  # pre-trained word embedding
    idf = pickle.load(open('../data/idf', 'rb'))  # pre-trained idf value of all words in the w2v dictionary

    test_query, test_answer, train_query, train_answer, test_feature, train_feature, rec_api_test, rec_api_train = split_data.get_test_train()
    AL_train_feature = get_AL_feature(train_answer, rec_api_train, train_feature)
    num_choose = 13
    top1, top3, top5, map, mrr = 0, 0, 0, 0, 0
    # iteration begin
    for round in range(10):
        choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose = split_data.split_choose_unlabel(train_query, train_answer, rec_api_train, num_choose)
        choose_feature, unlabel_feature = split_data.get_choose(AL_train_feature, choose)

        y_predict = get_AL_predict(test_feature, choose_feature, unlabel_feature, test_query, choose_query, choose_answer, unlabel_query, unlabel_answer, rec_api_test, rec_api_choose, rec_api_unlabel, w2v, idf)
        rank_mod, rankall = [], []
        for n in range(len(test_query)):
            temp_pred = y_predict[10 * n:10 * n + 10]
            pred, sum_pred = [], 0
            if sum_pred == 0:
                sum_pred =1
            for i in range(10):
                sum_pred += temp_pred[i]
            for num in range(10):
                sum = temp_pred[num]/5
                pred.append(sum)
            rank_mod, rankall = metric.re_sort(pred, rec_api_test, test_answer, n, rank_mod, rankall)
        temp_top1, temp_top3, temp_top5, temp_map, temp_mrr = metric.metric_val(rank_mod, rankall, len(rec_api_test))
        top1 += temp_top1
        top3 += temp_top3
        top5 += temp_top5
        map += temp_map
        mrr += temp_mrr
    print(top1/10, top3/10, top5/10, map/10, mrr/10)
