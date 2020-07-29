import split_data, feedback, metric
import xgboost as xgb
import gensim
import _pickle as pickle
import numpy as np
from sklearn.model_selection import KFold
import random
from sklearn.metrics import label_ranking_average_precision_score
import math
import csv
import time


#first LTR only
def get_LTR_feature(t_answer, t_rec_api, feature):
    training_feature = []
    for i, train in enumerate(t_answer):
        # print('train',i, train)
        for index, ap in enumerate(t_rec_api[i*10:i*10+10]):
            if ap in train:
                temp = [1]
            else:
                temp = [0]
            temp.extend(feature[i*10+index][1:])
            training_feature.append(temp)
            # print(temp)
    return training_feature


def evalerror(preds, dtrain):       # written by myself
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', label_ranking_average_precision_score(labels, preds)


def get_LTR_predict(test_feature, train_x_feature, train_y_feature):
    # 一共2组*每组3条，6条样本，特征维数是2
    n_group = int(len(train_x_feature)/10)
    n_testgroup = int(len(test_feature)/10)
    n_choice = 10

    # dtrain = np.random.uniform(0, 100, [n_group * n_choice, 2])
    # # numpy.random.choice(a, size=None, replace=True, p=None)
    # dtarget = np.array([np.random.choice([0, 1, 2], 3, False) for i in range(n_group)]).flatten()
    # # n_group用于表示从前到后每组各自有多少样本，前提是样本中各组是连续的，[3，3]表示一共6条样本中前3条是第一组，后3条是第二组
    dgroup = np.array([n_choice for i in range(n_group)]).flatten()
    print(len(train_x_feature), len(train_y_feature), len(dgroup), dgroup)

    dtrain = xgb.DMatrix(train_x_feature, train_y_feature)
    # dtrain.set_group(dgroup)
    num_rounds = 100
    plst = params.items()
    model = xgb.train(plst, dtrain, num_rounds)

    X_test = test_feature
    dtest = xgb.DMatrix(X_test)
    # dtestgroup = np.array([n_choice for i in range(n_testgroup)]).flatten()
    # dtest.set_group(dtestgroup)

    y_predict = model.predict(dtest)
    y_predict = y_predict.tolist()
    return y_predict


def get_LTR_predict_LTR(test_feature, test_feedback_info, choose_feature, train_feedback_info):
    X_train, y_train = split_data.get_train_feature_matrix(train_feedback_info, choose_feature)
    X_test = split_data.get_test_feature_matrix(test_feedback_info, test_feature)

    dtrain = xgb.DMatrix(X_train, y_train)
    num_rounds = 100
    plst = params.items()
    model = xgb.train(plst, dtrain, num_rounds)

    dtest = xgb.DMatrix(X_test)

    y_predict = model.predict(dtest)
    y_predict = y_predict.tolist()
    return y_predict


# params = {
#     'booster': 'gbtree',
#     'objective': 'rank:pairwise',
#     'eval_metric': 'map@5-',
#     'min_child_weight': 5,
#     'max_depth': 8,
#     'subsample': 0.5,
#     'colsample_bytree': 0.5,
#     'eta': 0.001,
#     'seed': 2020,
#     'nthread': -1,
#     'silent': True,
#     }
params = {
        'booster': 'gbtree',
        'objective': 'rank:pairwise',
        # 'objective': 'binary:logistic',
        'gamma': 1,
        'max_depth': 8,
        'lambda': 3,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'min_child_weight': 5,
        'silent': 1,
        'eta': 0.001,
        'seed': 2020,
        'alpha': 5,
        'nthread': -1,
        'eval_metric': 'map@5-',
    }
# params = {
#         'booster': 'gbtree',
#         'objective': 'rank:pairwise',
#         'gamma': 0.3,
#         'max_depth': 3,
#         'lambda': 1,
#         'subsample': 0.6,
#         'colsample_bytree': 0.8,
#         'min_child_weight': 3,
#         'silent': 1,
#         'eta': 0.01,
#         'seed': 100,
#         'alpha': 1,
#         'nthread': -1,
#         'eval_metric': 'map@5-',
#     }


if __name__=="__main__":
    # generate training dataset
    # 一共2组*每组3条，6条样本，特征维数是2
    n_group = 2
    n_choice = 3
    dtrain = np.random.uniform(0, 100, [n_group * n_choice, 2])
    # numpy.random.choice(a, size=None, replace=True, p=None)
    dtarget = np.array([np.random.choice([0, 1, 2], 3, False) for i in range(n_group)]).flatten()
    # n_group用于表示从前到后每组各自有多少样本，前提是样本中各组是连续的，[3，3]表示一共6条样本中前3条是第一组，后3条是第二组
    dgroup = np.array([n_choice for i in range(n_group)]).flatten()

    # concate Train data, very import here !
    xgbTrain = xgb.DMatrix(dtrain, label=dtarget)
    xgbTrain.set_group(dgroup)

    # generate eval data
    dtrain_eval = np.random.uniform(0, 100, [n_group * n_choice, 2])
    xgbTrain_eval = xgb.DMatrix(dtrain_eval, label=dtarget)
    xgbTrain_eval.set_group(dgroup)
    evallist = [(xgbTrain, 'train'), (xgbTrain_eval, 'eval')]

    # train model
    # xgb_rank_params1加上 evals 这个参数会报错，还没找到原因
    # rankModel = train(xgb_rank_params1,xgbTrain,num_boost_round=10)
    rankModel = xgb.train(params, xgbTrain, num_boost_round=20, evals=evallist)

    # test dataset
    dtest = np.random.uniform(0, 100, [n_group * n_choice, 2])
    dtestgroup = np.array([n_choice for i in range(n_group)]).flatten()
    xgbTest = xgb.DMatrix(dtest)
    xgbTest.set_group(dgroup)

    # test
    print(rankModel.predict(xgbTest))


    # start = time.time()
    # w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed')  # pre-trained word embedding
    # idf = pickle.load(open('../data/idf', 'rb'))  # pre-trained idf value of all words in the w2v dictionary
    #
    # # test_query, test_answer, train_query, train_answer, test_feature, train_feature, rec_api_test, rec_api_train = split_data.get_test_train()
    # # LTR_train_feature = get_LTR_feature(train_answer, rec_api_train, train_feature)
    # queries = []
    # fr = open('../data/feedback_all_original_biker.csv', 'r')
    # reader = csv.reader(fr)
    # for row in reader:
    #     queries.append(row[0])
    #
    # num_choose = 37
    # top1, top3, top5, map, mrr = 0, 0, 0, 0, 0
    # # iteration begin
    # for round in range(1):
    #     round_top1, round_top3, round_top5, round_map, round_mrr = 0, 0, 0, 0, 0
    #
    #     # k-fold cross validation
    #     kf = KFold(n_splits=10)
    #     for train_idx, test_idx in kf.split(queries):
    #         train_idx = list(train_idx)
    #         test_idx = list(test_idx)
    #         print('test_idx', test_idx)
    #
    #         # 数据分为训练集、反馈集、测试集
    #         choose_idx = sorted(random.sample(train_idx, num_choose))
    #         print('choose_idx', train_idx, len(train_idx), num_choose)
    #         train_idx = [i for i in train_idx if i not in choose_idx]
    #         pct = len(choose_idx) / (len(train_idx) + len(choose_idx))
    #         print(len(train_idx), len(test_idx), len(choose_idx), pct)
    #         print('---------------------')
    #
    #         # 获取测试数据
    #         test_query, test_answer, test_rec_api, test_feature = split_data.idx_to_data(test_idx)
    #
    #         # 获取AL初始训练数据，即反馈数据
    #         # choose_query, choose_answer, choose_rec_api, choose_feature = split_data.idx_to_data(choose_idx)
    #         choose_query, choose_answer, choose_rec_api, choose_feature = split_data.get_choose_data(choose_idx,
    #                                                                                                  test_query, pct,
    #                                                                                                  w2v, idf)
    #
    #         # 获取初始未标记数据，从stack overflow获取
    #         unlabel_query, unlabel_answer, unlabel_rec_api, unlabele_feature = split_data.get_unlabel_data(test_query,
    #                                                                                                        w2v, idf)
    #
    #         LTR_predict = get_LTR_predict(add_x_FV, add_x_FR, add_y_FR)
    #         train_x_FV, train_y_FV = split_data.get_train_feature_matrix(train_feedback_info, choose_feature)
    #         test_feature = np.array(test_feature)
    #
    #         y_predict = get_LTR_predict_LTR(test_feature, test_feedback_info, choose_feature, train_feedback_info)
    #         rank_mod, rankall = [], []
    #         for n in range(len(test_query)):
    #             temp_pred = y_predict[10 * n:10 * n + 10]
    #             pred, sum_pred = [], 0
    #             for i in range(10):
    #                 sum_pred += temp_pred[i]+5
    #             for num in range(10):
    #                 sum = (temp_pred[num]+5)/10
    #                 pred.append(sum)
    #             rank_mod, rankall = metric.re_sort(pred, rec_api_test, test_answer, n, rank_mod, rankall)
    #         temp_top1, temp_top3, temp_top5, temp_map, temp_mrr = metric.metric_val(rank_mod, rankall, len(rec_api_test))
    #         top1 += temp_top1
    #         top3 += temp_top3
    #         top5 += temp_top5
    #         map += temp_map
    #         mrr += temp_mrr
    # print(top1/10, top3/10, top5/10, map/10, mrr/10)
    #
    # fw = open('../data/metric_biker.csv', 'a+', newline='')
    # writer = csv.writer(fw)
    # writer.writerow(('LTR', num_choose, top1/10, top3/10, top5/10, map/10, mrr/10))
    # fw.close()
    #
    # end = time.time()
    # print(end-start)