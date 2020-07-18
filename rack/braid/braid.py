import split_data, feedback, metric, braid_LTR, braid_AL
import gensim
import _pickle as pickle
import csv
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import random
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

num_choose = 15

queries = []
fr = open('../data/feedback_all_new_rack.csv', 'r')
reader = csv.reader(fr)
for row in reader:
    queries.append(row[0])


# k-fold cross validation
kf = KFold(n_splits=10)
for train_idx, test_idx in kf.split(queries):
    train_idx = list(train_idx)
    test_idx = list(test_idx)
    print('test_idx', test_idx)

    # iteration for 10times because the feedback is chosen randomly
    # iteration begin
    for round in range(1):
        # 数据分为训练集、反馈集、测试集
        choose_idx = sorted(random.sample(train_idx, num_choose))
        train_idx = [i for i in train_idx if i not in choose_idx]
        print(len(train_idx), len(test_idx), len(choose_idx))
        print('---------------------')

        # 获取AL初始训练数据，即反馈数据
        choose_query, choose_answer, choose_rec_api, choose_feature = split_data.idx_to_data(choose_idx)
        AL_choose_feature = braid_AL.get_AL_feature(choose_answer, choose_rec_api, choose_feature)

        # 获取初始未标记数据，注意这里和之前结果的区别（将未标记数据集也做了0，1转化，之前也有），导致实验结果大幅度提升
        unlabel_query, unlabel_answer, unlabel_rec_api, unlabele_feature = split_data.idx_to_data(train_idx)
        AL_unlabel_feature = braid_AL.get_AL_feature(unlabel_answer, unlabel_rec_api, unlabele_feature)

        # 获取测试数据
        test_query, test_answer, test_rec_api, test_feature = split_data.idx_to_data(test_idx)

        # choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose = split_data.split_choose_unlabel(
        #     train_query, train_answer, rec_api_train, num_choose)
        # choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose = split_data.split_10_choose_unlabel(
        #     train_query, train_answer, rec_api_train)

        AL_predict, add_x_FV, add_x_FR, add_y_FR = braid_AL.get_AL_predict(test_feature, AL_choose_feature, AL_unlabel_feature, test_query, choose_query, choose_answer, unlabel_query, unlabel_answer, test_rec_api, choose_rec_api, unlabel_rec_api, w2v, idf)
        LTR_predict = braid_LTR.get_LTR_predict(add_x_FV, add_x_FR, add_y_FR)

        rank_mod, rankall, LTR_rank_mod, LTR_rankall, AL_rank_mod, AL_rankall = [], [], [], [], [], []
        m = 0.6
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
            rank_mod, rankall = metric.re_sort(pred, test_rec_api, test_answer, n, rank_mod, rankall)
            LTR_rank_mod, LTR_rankall = metric.ALTR_re_sort(LTR_pred, test_rec_api, test_answer, n, LTR_rank_mod, LTR_rankall)
            AL_rank_mod, AL_rankall = metric.ALTR_re_sort(AL_pred, test_rec_api, test_answer, n, AL_rank_mod, AL_rankall)
        temp_top1, temp_top3, temp_top5, temp_map, temp_mrr = metric.metric_val(rank_mod, rankall, len(test_rec_api))
        LTR_temp_top1, LTR_temp_top3, LTR_temp_top5, LTR_temp_map, LTR_temp_mrr = metric.metric_val(LTR_rank_mod, LTR_rankall, len(test_rec_api))
        AL_temp_top1, AL_temp_top3, AL_temp_top5, AL_temp_map, AL_temp_mrr = metric.metric_val(AL_rank_mod, AL_rankall, len(test_rec_api))
        top1 += temp_top1
        top3 += temp_top3
        top5 += temp_top5
        map += temp_map
        mrr += temp_mrr
        LTR_top1 += LTR_temp_top1
        LTR_top3 += LTR_temp_top3
        LTR_top5 += LTR_temp_top5
        LTR_map += LTR_temp_map
        LTR_mrr += LTR_temp_mrr
        AL_top1 += AL_temp_top1
        AL_top3 += AL_temp_top3
        AL_top5 += AL_temp_top5
        AL_map += AL_temp_map
        AL_mrr += AL_temp_mrr
print(top1/10, top3/10, top5/10, map/10, mrr/10)
print(LTR_top1/10, LTR_top3/10, LTR_top5/10, LTR_map/10, LTR_mrr/10)
print(AL_top1/10, AL_top3/10, AL_top5/10, AL_map/10, AL_mrr/10)

fw = open('../data/metric_nlp.csv', 'a+', newline='')
writer = csv.writer(fw)
writer.writerow(('BRAID', num_choose, top1/10, top3/10, top5/10, map/10, mrr/10))
writer.writerow(('LTR', num_choose, LTR_top1/10, LTR_top3/10, LTR_top5/10, LTR_map/10, LTR_mrr/10))
writer.writerow(('AL', num_choose, AL_top1/10, AL_top3/10, AL_top5/10, AL_map/10, AL_mrr/10))
# writer.writerow(('BRAID', '10_query', top1, top3, top5, map, mrr))
fw.close()

end = time.time()
print(end-start)
