import csv
import split_data, braid_AL, braid_LTR, metric, feedback
import xgboost as xgb
import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.linear_model import LogisticRegression
import random
import gensim
import _pickle as pickle
from os import path
import warnings
warnings.filterwarnings("ignore")

params = {
        'booster': 'gbtree',
        'objective': 'rank:pairwise',
        'gamma': 1,
        'eta': 0.001,
        'seed': 2020,
        'alpha': 3,
        'nthread': -1,
    }


def get_AL_predict(pct, test_feature, choose_feature, unlabel_feature, test_query, choose_query, choose_answer, unlabel_query, unlabel_answer, rec_api_test, rec_api_choose, rec_api_unlabel, w2v, idf):
    unlabel_feedback_info = feedback.get_feedback_inf(unlabel_query, choose_query, choose_answer, rec_api_unlabel, w2v, idf)
    label_feedback_info = feedback.get_feedback_inf(choose_query, choose_query, choose_answer, rec_api_choose, w2v, idf)
    X_train, y_train = braid_AL.get_active_data(unlabel_feedback_info, unlabel_feature)

    # pool-based sampling
    n_queries = int(pct*150)
    print('n_queries', n_queries, len(unlabel_query))
    for idx in range(n_queries):
        if len(unlabel_query) > 0:
            # add queried instance into FR
            choose_query.append(unlabel_query[idx])
            choose_answer.append(unlabel_answer[idx])
            rec_api_choose.extend(rec_api_unlabel[idx*10:idx*10+10])
            choose_feature.extend(unlabel_feature[idx*10:idx*10+10])

            # remove queried instance from pool
            for i in range(10):
                X_train = np.delete(X_train, idx*10, axis=0)
                y_train = np.delete(y_train, idx*10)
            del unlabel_query[idx]
            del unlabel_answer[idx]
            del rec_api_unlabel[idx*10:idx*10+10]
            del unlabel_feature[idx*10:idx*10+10]
            if len(X_train) == 0:
                break

    add_label_feedback_info = feedback.get_feedback_inf(choose_query, choose_query, choose_answer, rec_api_choose, w2v, idf)
    new_X_feedback, new_y_feedback = braid_AL.get_active_data(add_label_feedback_info, choose_feature)
    feedback_info = feedback.get_feedback_inf(test_query, choose_query, choose_answer, rec_api_test, w2v, idf)
    X = split_data.get_test_feature_matrix(feedback_info, test_feature)
    print('new_X_feedback', len(new_X_feedback), len(new_y_feedback))

    return X, new_X_feedback, new_y_feedback, choose_query, choose_answer, rec_api_choose, choose_feature


path1 = 'D:/first_review_data/'

w2v = gensim.models.Word2Vec.load(path.join(path1, './w2v_model_stemmed'))  # pre-trained word embedding
idf = pickle.load(open(path.join(path1, './idf'), 'rb'))  # pre-trained idf value of all words in the w2v dictionary

for round in range(10):
    top1, top3, top5, map, mrr = 0, 0, 0, 0, 0
    LTR_top1, LTR_top3, LTR_top5, LTR_map, LTR_mrr = 0, 0, 0, 0, 0
    AL_top1, AL_top3, AL_top5, AL_map, AL_mrr = 0, 0, 0, 0, 0
    X_train, y_train = [], []

    dataset_size = 150
    # filter_idx = split_data.filter_test_idx('nlp')
    filter_idx = [i for i in range(dataset_size)]
    test_idx = sorted(random.sample(filter_idx, 50))
    print('len(test_idx)', len(test_idx), test_idx)
    test_query, test_answer, test_rec_api, test_feature = split_data.idx_to_data(test_idx)
    # train_idx = [i for i in range(dataset_size) if i not in test_idx]
    # 初始feedback repository为空
    choose_query, choose_answer, choose_rec_api, choose_feature = [], [], [], []
    print('len(test_query)', len(test_query))

    # 反馈数据库
    for id in range(len(test_query)):
        round_top1, round_top3, round_top5, round_map, round_mrr = 0, 0, 0, 0, 0
        LTR_round_top1, LTR_round_top3, LTR_round_top5, LTR_round_map, LTR_round_mrr = 0, 0, 0, 0, 0
        AL_round_top1, AL_round_top3, AL_round_top5, AL_round_map, AL_round_mrr = 0, 0, 0, 0, 0

        unlabel_query, unlabel_answer, unlabel_rec_api, unlabel_feature = [], [], [], []
        tmp_unlabel_query, tmp_unlabel_answer, tmp_unlabel_rec_api, tmp_unlabel_feature = split_data.get_unlabel_data([test_query[id]], w2v, idf)

        # 过滤掉已加入feedback repository的unlabel数据
        filter_idx = []
        for q in tmp_unlabel_query:
            if q in choose_query and tmp_unlabel_query.index(q) not in filter_idx:
                filter_idx.append(tmp_unlabel_query.index(q))
                # print(q)
                # print(choose_query.index(q), tmp_unlabel_query.index(q))
        print('filter_idx', len(filter_idx), filter_idx, len(choose_query))
        for i in range(len(tmp_unlabel_query)):
            if i not in filter_idx:
                unlabel_query.append(tmp_unlabel_query[i])
                unlabel_answer.append(tmp_unlabel_answer[i])
                unlabel_rec_api.extend(tmp_unlabel_rec_api[i*10:i*10+10])
                unlabel_feature.extend(tmp_unlabel_feature[i*10:i*10+10])

        # 训练模型：
        if id == 0:
            unlabel_feedback_info = feedback.get_feedback_inf(unlabel_query, choose_query, choose_answer, unlabel_rec_api, w2v, idf)
            label_feedback_info = feedback.get_feedback_inf(choose_query, choose_query, choose_answer, choose_rec_api, w2v, idf)
            X_train, y_train = braid_AL.get_active_data(unlabel_feedback_info, unlabel_feature)
            X_feedback, y_feedback = braid_AL.get_active_data(label_feedback_info, choose_feature)

            predict = []
            plst = params.items()
            print('len', len(X_feedback), len(unlabel_query))
            if len(X_feedback) > 0:
                # initializing the active learner
                learner = ActiveLearner(
                    estimator=LogisticRegression(penalty='l1', solver='liblinear'),
                    X_training=X_feedback, y_training=y_feedback
                )
                dtrain = xgb.DMatrix(X_feedback, y_feedback)
                model = xgb.train(plst, dtrain, 100)
            else:
                learner = ActiveLearner(
                    estimator=LogisticRegression(penalty='l1', solver='liblinear'),
                    X_training=X_train, y_training=y_train
                )
                dtrain = xgb.DMatrix(X_train, y_train)
                model = xgb.train(plst, dtrain, 100)
        elif len(unlabel_query) > 0:
            # pool-based sampling
            n_queries = 20
            print('n_queries', n_queries, len(unlabel_query))
            for idx in range(n_queries):
                # add queried instance into FR
                if idx < len(unlabel_query):
                    choose_query.append(unlabel_query[idx])
                    choose_answer.append(unlabel_answer[idx])
                    choose_rec_api.extend(unlabel_rec_api[idx * 10:idx * 10 + 10])
                    choose_feature.extend(unlabel_feature[idx * 10:idx * 10 + 10])

        # if id == 25:
        #     label_feedback_info = feedback.get_feedback_inf(choose_query, choose_query, choose_answer, choose_rec_api, w2v, idf)
        #     X_feedback, y_feedback = braid_AL.get_active_data(label_feedback_info, choose_feature)
        #     learner = ActiveLearner(
        #         estimator=LogisticRegression(penalty='l1', solver='liblinear'),
        #         X_training=X_feedback, y_training=y_feedback
        #     )
        #     dtrain = xgb.DMatrix(X_feedback, y_feedback)
        #     model = xgb.train(plst, dtrain, 100)

        # AL预测模型
        print('filter_idx', len(filter_idx), filter_idx, len(choose_query))
        AL_predict = []
        feedback_info = feedback.get_feedback_inf([test_query[id]], choose_query, choose_answer, test_rec_api[id*10:id*10+10], w2v, idf)
        X = split_data.get_test_feature_matrix(feedback_info, test_feature[id*10:id*10+10])
        X_test = np.array(X)
        # 用反馈数据学习过后的模型来预测测试数据
        for query_idx in range(len(X)):
            y_pre = learner.predict_proba(X=X_test[query_idx].reshape(1, -1))
            AL_predict.append(float(y_pre[0, 1]))

        # LTR预测模型
        dtest = xgb.DMatrix(X)
        y_predict = model.predict(dtest)
        LTR_predict = y_predict.tolist()
        # LTR_predict = braid_LTR.get_LTR_predict(add_x_FV, add_x_FR, add_y_FR)

        print('len(query)', len(unlabel_query), len(choose_query))
        choose_query.append(test_query[id])
        choose_answer.append(test_answer[id])
        choose_rec_api.extend(test_rec_api[id*10:id*10+10])
        choose_feature.extend(test_feature[id*10:id*10+10])

        rank_mod, rankall, LTR_rank_mod, LTR_rankall, AL_rank_mod, AL_rankall = [], [], [], [], [], []
        m = 0.1
        n = 0
        # for n in range(len(test_query)):
        pred1 = LTR_predict[10*n:10*n+10]
        pred2 = AL_predict[10*n:10*n+10]
        pred, sum_pred1,sum_pred2 = [], 0, 0
        LTR_pred, AL_pred, LTR_sum,AL_sum = [], [], 0, 0
        for i in range(10):
            sum_pred1 += pred1[i] + 5
            sum_pred2 += pred2[i]
        al_idx = []
        rerank_al = sorted(pred2, reverse=True)
        for i in range(10):
            temp = rerank_al.index(pred2[i])+1
            while temp in al_idx:
                temp += 1
            al_idx.append(temp)
        print(al_idx)
        for num in range(10):
            sum = (pred1[num]+5)/10+m*pred2[num]/al_idx[num]
            LTR_sum = (pred1[num]+5)/10
            AL_sum = m*pred2[num]/al_idx[num]
            pred.append(sum)
            LTR_pred.append(LTR_sum)
            AL_pred.append(AL_sum)
        print(LTR_pred)
        print(AL_pred)
        print(test_rec_api[10*id:10*id+10])
        # fr_rec_score, fr_rec_api = split_data.get_fr_cal_sim(test_query[n], fr_matrix, fr_idf_vector, fr_answers, pct, w2v, idf)
        rank_mod, rankall = metric.re_sort([test_query[id]], pred, test_rec_api[id*10:id*10+10], [test_answer[id]], n, rank_mod, rankall)
        LTR_rank_mod, LTR_rankall = metric.ALTR_re_sort(LTR_pred, test_rec_api[id*10:id*10+10], [test_answer[id]], n, LTR_rank_mod, LTR_rankall)
        AL_rank_mod, AL_rankall = metric.ALTR_re_sort(AL_pred, test_rec_api[id*10:id*10+10], [test_answer[id]], n, AL_rank_mod, AL_rankall)

        temp_top1, temp_top3, temp_top5, temp_map, temp_mrr = metric.metric_val(rank_mod, rankall, len(test_rec_api[id*10:id*10+10]))
        LTR_temp_top1, LTR_temp_top3, LTR_temp_top5, LTR_temp_map, LTR_temp_mrr = metric.metric_val(LTR_rank_mod, LTR_rankall, len(test_rec_api[id*10:id*10+10]))
        AL_temp_top1, AL_temp_top3, AL_temp_top5, AL_temp_map, AL_temp_mrr = metric.metric_val(AL_rank_mod, AL_rankall, len(test_rec_api[id*10:id*10+10]))
        round_top1 += temp_top1
        round_top3 += temp_top3
        round_top5 += temp_top5
        round_map += temp_map
        round_mrr += temp_mrr
        top1 += temp_top1
        top3 += temp_top3
        top5 += temp_top5
        map += temp_map
        mrr += temp_mrr
        LTR_round_top1 += LTR_temp_top1
        LTR_round_top3 += LTR_temp_top3
        LTR_round_top5 += LTR_temp_top5
        LTR_round_map += LTR_temp_map
        LTR_round_mrr += LTR_temp_mrr
        LTR_top1 += LTR_temp_top1
        LTR_top3 += LTR_temp_top3
        LTR_top5 += LTR_temp_top5
        LTR_map += LTR_temp_map
        LTR_mrr += LTR_temp_mrr
        AL_round_top1 += AL_temp_top1
        AL_round_top3 += AL_temp_top3
        AL_round_top5 += AL_temp_top5
        AL_round_map += AL_temp_map
        AL_round_mrr += AL_temp_mrr
        AL_top1 += AL_temp_top1
        AL_top3 += AL_temp_top3
        AL_top5 += AL_temp_top5
        AL_map += AL_temp_map
        AL_mrr += AL_temp_mrr

        # fw = open('../data/metric_biker.csv', 'a+', newline='')
        # writer = csv.writer(fw)
        # writer.writerow(('BRAID', id+1, top1/(id+1), top3/(id+1), top5/(id+1), map/(id+1), mrr/(id+1)))
        # fw.close()

        print(top1/(id+1), top3/(id+1), top5/(id+1), map/(id+1), mrr/(id+1))

    print(top1/50, top3/50, top5/50, map/50, mrr/50)
    print(LTR_top1/50, LTR_top3/50, LTR_top5/50, LTR_map/50, LTR_mrr/50)
    print(AL_top1/50, AL_top3/50, AL_top5/50, AL_map/50, AL_mrr/50)

    fw = open('../data/metric_rack.csv', 'a+', newline='')
    writer = csv.writer(fw)
    writer.writerow(('BRAID', top1/50, top3/50, top5/50, map/50, mrr/50))
    fw.close()
