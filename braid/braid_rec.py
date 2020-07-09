import csv
import math
from preprocess import recommendation
from preprocess import similarity
from main import APIRec, FeedbackInfo, FeatureVector
import gensim
import _pickle as pickle
import time
import xgboost as xgb
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import split_data, feedback, metric, braid_LTR, braid_AL
from os import path
import warnings
warnings.filterwarnings("ignore")





def text2feat(api, api_descriptions, w2v, idf, query_matrix, query_idf_vector):
    api_matrix, api_idf_vector = feedback.load_matrix(api, w2v, idf)
    api_descriptions_matrix = similarity.init_doc_matrix(api_descriptions, w2v)
    api_descriptions_idf_vector = similarity.init_doc_idf_vector(api_descriptions, idf)

    # 获取api及doc信息并计算其相似度，相关问题在推荐中已经获得
    api_sim = similarity.sim_doc_pair(query_matrix, api_matrix, query_idf_vector, api_idf_vector)
    if api_descriptions == []:
        api_desc_sim = 0
    else:
        api_desc_sim = similarity.sim_doc_pair(query_matrix, api_descriptions_matrix, query_idf_vector, api_descriptions_idf_vector)

    # 将获得信息按api为一列放入sum_inf中
    sum_inf = list()
    sum_inf.append(api_sim)
    sum_inf.append(api_desc_sim)

    return sum_inf


def get_AL_predict(test_feature, choose_feature, unlabel_feature, test_query, choose_pair, unlabel_pair, rec_api_test, w2v, idf):
    unlabel_feature_copy = unlabel_feature.copy()
    feedback.get_feedback_inf(unlabel_pair, choose_pair, unlabel_feature_copy, w2v, idf)
    feedback.get_feedback_inf(choose_pair, choose_pair, choose_feature, w2v, idf)
    X_train, y_train = braid_AL.get_active_data(unlabel_feature_copy)
    X_feedback, y_feedback = braid_AL.get_active_data(choose_feature)

    # initializing the active learner
    learner = ActiveLearner(
        estimator=KNeighborsClassifier(n_neighbors=4),
        # estimator=LogisticRegression(penalty='l1', solver='liblinear'),
        X_training=X_feedback, y_training=y_feedback
    )

    length = len(rec_api_test)
    predict, sel_query, add_unlabel_feature = [], [], []
    if len(unlabel_pair) > 0:
        # pool-based sampling
        n_queries = 40
        for idx in range(n_queries):
            query_idx, query_instance = uncertainty_sampling(classifier=learner, X=X_train)
            idx = int(query_idx/10)
            learner.teach(
                X=X_train[query_idx].reshape(1, -1),
                y=y_train[query_idx].reshape(1, )
            )

            # add queried instance into FR
            choose_pair.append(FeedbackInfo(unlabel_pair[idx].query, unlabel_pair[idx].api))
            for i in range(idx*10, idx*10+10):
                choose_feature.append(FeatureVector(unlabel_feature[i].label, unlabel_feature[i].feature, unlabel_feature[i].title))

            # remove queried instance from pool
            for i in range(10):
                X_train = np.delete(X_train, idx*10, axis=0)
                y_train = np.delete(y_train, idx*10)
            del unlabel_pair[idx]
            for i in range(idx*10, idx*10+10):
                del unlabel_feature[i]
            if len(X_train) == 0:
                break

    feedback.get_feedback_inf(choose_pair, choose_pair, choose_feature, w2v, idf)
    new_X_feedback, new_y_feedback = braid_AL.get_active_data(choose_feature)
    learner = ActiveLearner(
        estimator=KNeighborsClassifier(n_neighbors=4),
        # estimator=LogisticRegression(penalty='l1', solver='liblinear'),
        X_training=new_X_feedback, y_training=new_y_feedback
    )
    feedback.get_feedback_inf(test_query, choose_pair, rec_api_test, w2v, idf)
    feedbaci_inf = [row.feedback_sim for row in rec_api_test]
    X = split_data.get_test_feature_matrix(feedbaci_inf, test_feature)

    X_test = np.array(X)
    # 用反馈数据学习过后的模型来预测测试数据
    for query_idx in range(length):
        try:
            y_pre = learner.predict_proba(X=X_test[query_idx].reshape(1, -1))
        except ValueError:
            predict = [0.0 for n in range(length)]
            # print('Input the error query')
        else:
            predict.append(float(y_pre[0, 1]))

    return predict, X, new_X_feedback, new_y_feedback

if __name__=='__main__':
    path1 = 'I:/first_review_biker/first_review_data/'

    w2v = gensim.models.Word2Vec.load(path.join(path1, './w2v_model_stemmed'))  # pre-trained word embedding
    idf = pickle.load(open(path.join(path1, './idf'), 'rb'))  # pre-trained idf value of all words in the w2v dictionary
    questions = pickle.load(open(path.join(path1, './api_questions_pickle_new'), 'rb'))  # the pre-trained knowledge base of api-related questions (about 120K questions)
    questions = recommendation.preprocess_all_questions(questions, idf, w2v)  # matrix transformation
    javadoc = pickle.load(
        open(path.join(path1, './javadoc_pickle_wordsegmented'), 'rb'))  # the pre-trained knowledge base of javadoc
    javadoc_dict_classes = dict()
    javadoc_dict_methods = dict()
    recommendation.preprocess_javadoc(javadoc, javadoc_dict_classes, javadoc_dict_methods, idf,
                                      w2v)  # matrix transformation
    parent = dict()  # In online mode, there is no need to remove duplicate question of the query

    print('loading data finished')

    # feedback info of user query from FR
    fr = open(path.join(path1, './feedback_rec.csv'), 'r')
    reader = csv.reader(fr)
    choose_pair = []
    for row in reader:
        print(row[0], row[1:])

    while True:
        print('Please input your query:')
        query = input()
        # query = 'how to convert int to string?'
        if not query:
            continue
        query_matrix, query_idf_vector = feedback.load_matrix(query, w2v, idf)

        top_questions = recommendation.get_topk_questions(query, query_matrix, query_idf_vector, questions, 50, parent)
        # recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions, javadoc,javadoc_dict_methods,-1)
        recommended_api = recommendation.recommend_api_class(query_matrix, query_idf_vector, top_questions, questions, javadoc,javadoc_dict_classes,-1)


        # combine api_relevant feature with FF
        pos = -1
        rec_api = []
        x = []
        for i,api in enumerate(recommended_api):
            print('Rank',i+1,':',api)
            # api_description, questions_titles = recommendation.summarize_api_method(api, top_questions, questions, javadoc,
            #                                                                          javadoc_dict_methods)
            api_description, questions_titles = recommendation.summarize_api_class(api, top_questions, questions, javadoc,
                                                                                     javadoc_dict_classes)

            sum_inf = text2feat(api, api_description, w2v, idf, query_matrix, query_idf_vector)

            rec_api.append(APIRec(i, api, api_description))
            rec_api[i].api_relate_sim = sum_inf

            if i == 9:
                break
        # print('##################')

        start1 = time.time()
        # feedback info of user query from SO
        fr = open(path.join(path1, './feedback_all.csv'), 'r')
        reader = csv.reader(fr)
        so_pair = []
        for row in reader:
            so_pair.append(FeedbackInfo(row[0], row[1:]))

        # feedback info of user query from FR
        fr = open(path.join(path1, './feedback_rec.csv'), 'r')
        reader = csv.reader(fr)
        choose_pair = []
        for row in reader:
            choose_pair.append(FeedbackInfo(row[0], row[1:]))
        feedback.get_feedback_inf(query, choose_pair, rec_api, w2v, idf)

        # FV = RF+FF
        for i in range(len(rec_api)):
            sum = rec_api[i].api_relate_sim
            sum.extend(rec_api[i].feedback_sim)
            x.append(sum)

        # feature info of FR
        fr = open(path.join(path1, './feedback_feature_rec.csv'), 'r')
        reader = csv.reader(fr)
        fr_feature = []
        for row in reader:
            fr_feature.append(FeatureVector(row[0], row[1:-1], row[-1]))

        #feature info of SO
        fr = open(path.join(path1, './get_feature_method.csv'), 'r')
        reader = csv.reader(fr)
        so_feature = []
        for row in reader:
            so_feature.append(FeatureVector(row[0], row[1:-1], row[-1]))

        pred2, add_x_FR, add_x_FV, add_y_FV = get_AL_predict(x, fr_feature, so_feature, query, choose_pair, so_pair, rec_api, w2v, idf)

        pred1 = braid_LTR.get_LTR_predict(add_x_FR, add_x_FV, add_y_FV)
        print(pred1)
        print('-----')
        print(pred2)

        rem = -10
        for api in rec_api:
            api.pred_LTR = pred1[api.init_id]+5
            api.pred_AL = pred2[api.init_id]

        # 获取pred_AL排序:pos_i
        al_idx = []
        for i in sorted(pred2, reverse=True):
            al_idx.append(pred2.index(i)+1)
            pred2[pred2.index(i)] = -1
        print('rerank_AL', al_idx)

        # 计算API_pred
        m = 0.6
        pred = []
        for api in rec_api:
            api.pred = (pred1[api.init_id]+5)/len(x) + m*pred2[api.init_id]/al_idx[api.init_id]
            pred.append(api.pred)
        print(pred)

        sort = []
        for i in range(len(rec_api)):
            sort.append(pred.index(max(pred)) + 1)
            pred[pred.index(max(pred))] = rem
        print(sort, rec_api)

        # 将api重新排序，输出相关结果
        responseToClient = []
        for i in sort:
            print(sort.index(i) + 1, rec_api[i-1].title)
            api_obj = {'id': sort.index(i) + 1, 'api': rec_api[i-1].title}
            responseToClient.append(api_obj)
        # start5 = time.time()

        print(responseToClient)

        print('choose API:')
        choose = input()

        if not math.isnan(rec_api[0].api_relate_sim[0]):
            if int(choose):
                fw = open(path.join(path1, './feedback_rec.csv', 'a+'), newline='')
                writer = csv.writer(fw)
                writer.writerow((query, rec_api[sort[int(choose)-1]-1].title))
                fw.close()
                fw = open(path.join(path1, './feedback_feature_rec.csv'), 'a+', newline='')
                writer = csv.writer(fw)
                for i in sort:
                    y = 0
                    if sort.index(i) == int(choose)-1:
                        y = 1
                        rec_api[i-1].feedback_sim[0] = 1
                    writer.writerow([y] + rec_api[i-1].api_relate_sim[:2] + rec_api[i-1].feedback_sim + [rec_api[i-1].title])
                n = len(rec_api)
                while n < 10:
                    writer.writerow((0, 0, 0, 0, 0, 0, 0, 0, 'null'))
                    n += 1
                fw.close()
                print(query, rec_api[sort[int(choose)-1]-1].title)
            else:
                print('none')

