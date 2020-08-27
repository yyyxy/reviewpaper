import split_data, feedback, metric, braid_LTR, braid_AL
import gensim
import _pickle as pickle
import csv
import time
from sklearn.model_selection import KFold
import random
import numpy as np
from os import path
import warnings
warnings.filterwarnings("ignore")

start = time.time()
path1 = 'D:/first_review_data/'

w2v = gensim.models.Word2Vec.load(path.join(path1, './w2v_model_stemmed'))  # pre-trained word embedding
idf = pickle.load(open(path.join(path1, './idf'), 'rb'))  # pre-trained idf value of all words in the w2v dictionary


top1, top3, top5, map, mrr = 0, 0, 0, 0, 0
LTR_top1, LTR_top3, LTR_top5, LTR_map, LTR_mrr = 0, 0, 0, 0, 0
AL_top1, AL_top3, AL_top5, AL_map, AL_mrr = 0, 0, 0, 0, 0

num_choose = 10
file_choose = 19

queries = []
fr = open('../data/feedback_all_new_nlp.csv', 'r')
reader = csv.reader(fr)
for row in reader:
    queries.append(row[0])

# # 增加query和so query的相似度计算作为分数组成部分
# # 索引转化为数据
# fr = open('../data/feedback_repository_rack_sim7.csv', 'r')
# reader = csv.reader(fr)
# fr_queries, fr_answers = [], []
# fr_matrix, fr_idf_vector = [], []
# for row in reader:
#     fr_queries.append(row[0])
#     fr_answers.append(row[1])
#
#     q1_matrix, q1_idf_vector = feedback.load_matrix(row[0], w2v, idf)
#     fr_matrix.append(q1_matrix)
#     fr_idf_vector.append(q1_idf_vector)

# fw = open('../data/biker_rank.csv', 'w', newline='')
# writer = csv.writer(fw)
# writer.writerow(('query', 'answer', 'rec_api', 'rank', 'original_rank'))
# fw.close()

# iteration for 10times because the feedback is chosen randomly
# iteration begin
for round in range(1):
    round_top1, round_top3, round_top5, round_map, round_mrr = 0, 0, 0, 0, 0
    LTR_round_top1, LTR_round_top3, LTR_round_top5, LTR_round_map, LTR_round_mrr = 0, 0, 0, 0, 0
    AL_round_top1, AL_round_top3, AL_round_top5, AL_round_map, AL_round_mrr = 0, 0, 0, 0, 0

    # k-fold cross validation
    kf = KFold(n_splits=10)
    for train_idx, test_idx in kf.split(queries):
        train_idx = list(train_idx)
        test_idx = list(test_idx)
        print('test_idx', len(test_idx), test_idx)

        # 数据分为训练集、反馈集、测试集
        choose_idx = sorted(random.sample(train_idx, num_choose))
        print('choose_idx', train_idx, len(train_idx), num_choose)
        train_idx = [i for i in train_idx if i not in choose_idx]
        pct = len(choose_idx)/(len(train_idx)+len(choose_idx))
        print(len(train_idx), len(test_idx), len(choose_idx), pct)
        print('---------------------')

        # 获取测试数据
        test_query, test_answer, test_rec_api, test_feature = split_data.idx_to_data(test_idx)
        time_ex1 = time.time()
        # 获取AL初始训练数据，即反馈数据
        # choose_query, choose_answer, choose_rec_api, choose_feature = split_data.idx_to_data(choose_idx)
        # choose_query, choose_answer, choose_rec_api, choose_feature = split_data.get_choose_data(choose_idx, test_query, pct, w2v, idf)
        choose_query, choose_answer, choose_rec_api, choose_feature = split_data.split_10_choose_data(file_choose)
        AL_choose_feature = braid_AL.get_AL_feature(choose_answer, choose_rec_api, choose_feature)

        # 获取初始未标记数据，从stack overflow获取
        # unlabel_query, unlabel_answer, unlabel_rec_api, unlabele_feature = split_data.get_choose_data(train_idx, test_query, pct, w2v, idf)
        unlabel_query, unlabel_answer, unlabel_rec_api, unlabele_feature = split_data.get_unlabel_data(test_query, w2v, idf)
        AL_unlabel_feature = braid_AL.get_AL_feature(unlabel_answer, unlabel_rec_api, unlabele_feature)
        time_ex2 = time.time()

        # choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose = split_data.split_choose_unlabel(
        #     train_query, train_answer, rec_api_train, num_choose)
        # choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose = split_data.split_10_choose_unlabel(
        #     train_query, train_answer, rec_api_train)
        time_train1 = time.time()
        AL_predict, add_x_FV, add_x_FR, add_y_FR = braid_AL.get_AL_predict(pct, test_feature, AL_choose_feature, AL_unlabel_feature, test_query, choose_query, choose_answer, unlabel_query, unlabel_answer, test_rec_api, choose_rec_api, unlabel_rec_api, w2v, idf)
        LTR_predict = braid_LTR.get_LTR_predict(add_x_FV, add_x_FR, add_y_FR)
        time_train2 = time.time()

        time_rerank1 = time.time()
        rank_mod, rankall, LTR_rank_mod, LTR_rankall, AL_rank_mod, AL_rankall = [], [], [], [], [], []
        m = 0.1
        for n in range(len(test_query)):
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
            print(test_rec_api[10*n:10*n+10])
            # fr_rec_score, fr_rec_api = split_data.get_fr_cal_sim(test_query[n], fr_matrix, fr_idf_vector, fr_answers, pct, w2v, idf)
            rank_mod, rankall = metric.re_sort(test_query, pred, test_rec_api, test_answer, n, rank_mod, rankall)
            LTR_rank_mod, LTR_rankall = metric.ALTR_re_sort(LTR_pred, test_rec_api, test_answer, n, LTR_rank_mod, LTR_rankall)
            AL_rank_mod, AL_rankall = metric.ALTR_re_sort(AL_pred, test_rec_api, test_answer, n, AL_rank_mod, AL_rankall)
        temp_top1, temp_top3, temp_top5, temp_map, temp_mrr = metric.metric_val(rank_mod, rankall, len(test_rec_api))
        LTR_temp_top1, LTR_temp_top3, LTR_temp_top5, LTR_temp_map, LTR_temp_mrr = metric.metric_val(LTR_rank_mod, LTR_rankall, len(test_rec_api))
        AL_temp_top1, AL_temp_top3, AL_temp_top5, AL_temp_map, AL_temp_mrr = metric.metric_val(AL_rank_mod, AL_rankall, len(test_rec_api))
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
        time_rerank2 = time.time()

        # fw = open('../data/runtime.csv', 'a+', newline='')
        # writer = csv.writer(fw)
        # writer.writerow((len(test_idx), num_choose, time_ex2-time_ex1, time_train2-time_train1, time_rerank2-time_rerank1, time_rerank2-time_ex1))
        # print(len(test_idx), time_ex2-time_ex1, time_train2-time_train1, time_rerank2-time_rerank1, time_rerank2-time_ex1)
        # fw.close()

    # fw = open('../data/metric_biker.csv', 'a+', newline='')
    # writer = csv.writer(fw)
    # writer.writerow(('BRAID', round+1, num_choose, round_top1/10, round_top3/10, round_top5/10, round_map/10, round_mrr/10))
    # writer.writerow(('LTR', round+1, num_choose, LTR_round_top1/10, LTR_round_top3/10, LTR_round_top5/10, LTR_round_map/10, LTR_round_mrr/10))
    # writer.writerow(('AL', round+1, num_choose, AL_round_top1/10, AL_round_top3/10, AL_round_top5/10, AL_round_map/10, AL_round_mrr/10))
    # fw.close()

print(top1/100, top3/100, top5/100, map/100, mrr/100)
print(LTR_top1/100, LTR_top3/100, LTR_top5/100, LTR_map/100, LTR_mrr/100)
print(AL_top1/100, AL_top3/100, AL_top5/100, AL_map/100, AL_mrr/100)

fw = open('../data/metric_nlp.csv', 'a+', newline='')
writer = csv.writer(fw)
# writer.writerow(('BRAID', 11, num_choose, top1/100, top3/100, top5/100, map/100, mrr/100))
# writer.writerow(('LTR', 11, num_choose, LTR_top1/100, LTR_top3/100, LTR_top5/100, LTR_map/100, LTR_mrr/100))
# writer.writerow(('AL', 11, num_choose, AL_top1/100, AL_top3/100, AL_top5/100, AL_map/100, AL_mrr/100))
writer.writerow(('BRAID', '10_query', file_choose, top1/10, top3/10, top5/10, map/10, mrr/10))
fw.close()

end = time.time()
print(end-start)
