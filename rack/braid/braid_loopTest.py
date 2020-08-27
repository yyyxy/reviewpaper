import csv
import split_data, braid_AL, braid_LTR, metric
import random
import gensim
import _pickle as pickle
from os import path
import warnings
warnings.filterwarnings("ignore")

path1 = 'D:/first_review_data/'


w2v = gensim.models.Word2Vec.load(path.join(path1, './w2v_model_stemmed'))  # pre-trained word embedding
idf = pickle.load(open(path.join(path1, './idf'), 'rb'))  # pre-trained idf value of all words in the w2v dictionary

top1, top3, top5, map, mrr = 0, 0, 0, 0, 0
LTR_top1, LTR_top3, LTR_top5, LTR_map, LTR_mrr = 0, 0, 0, 0, 0
AL_top1, AL_top3, AL_top5, AL_map, AL_mrr = 0, 0, 0, 0, 0

dataset_size = 413
filter_idx = split_data.filter_test_idx('biker')
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

    AL_choose_feature = braid_AL.get_AL_feature(choose_answer, choose_rec_api, choose_feature)

    unlabel_query, unlabel_answer, unlabel_rec_api, unlabele_feature = [], [], [], []
    tmp_unlabel_query, tmp_unlabel_answer, tmp_unlabel_rec_api, tmp_unlabele_feature = split_data.get_unlabel_data([test_query[id]], w2v, idf)
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
            unlabele_feature.extend(tmp_unlabele_feature[i*10:i*10+10])
    AL_unlabel_feature = braid_AL.get_AL_feature(unlabel_answer, unlabel_rec_api, unlabele_feature)

    # pct = (id+1)/(dataset_size-50)
    AL_predict, add_x_FV, add_x_FR, add_y_FR, choose_query, choose_answer, choose_rec_api, choose_feature = braid_AL.get_AL_predict(1/15, test_feature[id*10:id*10+10], AL_choose_feature,
                                                                       AL_unlabel_feature, [test_query[id]], choose_query,
                                                                       choose_answer, unlabel_query, unlabel_answer,
                                                                       test_rec_api[id*10:id*10+10], choose_rec_api, unlabel_rec_api,
                                                                       w2v, idf)
    LTR_predict = braid_LTR.get_LTR_predict(add_x_FV, add_x_FR, add_y_FR)
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

    fw = open('../data/metric_biker.csv', 'a+', newline='')
    writer = csv.writer(fw)
    writer.writerow(('BRAID', id+1, top1/(id+1), top3/(id+1), top5/(id+1), map/(id+1), mrr/(id+1)))
    fw.close()

    print(top1/(id+1), top3/(id+1), top5/(id+1), map/(id+1), mrr/(id+1))

print(top1/50, top3/50, top5/50, map/50, mrr/50)
print(LTR_top1/50, LTR_top3/50, LTR_top5/50, LTR_map/50, LTR_mrr/50)
print(AL_top1/50, AL_top3/50, AL_top5/50, AL_map/50, AL_mrr/50)

fw = open('../data/metric_biker.csv', 'a+', newline='')
writer = csv.writer(fw)
writer.writerow(('BRAID', top1/50, top3/50, top5/50, map/50, mrr/50))
# writer.writerow(('LTR', LTR_top1/50, LTR_top3/50, LTR_top5/50, LTR_map/50, LTR_mrr/50))
# writer.writerow(('AL', AL_top1/50, AL_top3/50, AL_top5/50, AL_map/50, AL_mrr/50))
fw.close()
