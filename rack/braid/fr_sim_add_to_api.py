import csv
import random
import feedback
from preprocess import similarity
import gensim
import _pickle as pickle
from os import path

def get_fr_cal_sim():
    path1 = 'D:/first_review_data/'
    w2v = gensim.models.Word2Vec.load(path.join(path1, './w2v_model_stemmed'))  # pre-trained word embedding
    idf = pickle.load(open(path.join(path1, './idf'), 'rb'))  # pre-trained idf value of all words in the w2v dictionary


    # 索引转化为数据
    fr = open('../data/feedback_repository_biker_sim8.csv', 'r')
    reader = csv.reader(fr)
    queries, answers = [], []
    q_matrix, q_idf_vector = [], []
    for row in reader:
        queries.append(row[0])
        answers.append(row[1])

        q1_matrix, q1_idf_vector = feedback.load_matrix(row[0], w2v, idf)
        q_matrix.append(q1_matrix)
        q_idf_vector.append(q1_idf_vector)

    rec_api, rec_score = [], []
    fr = open('../data/feedback_all_original_biker.csv', 'r')
    reader = csv.reader(fr)
    for row in reader:
        query_matrix, query_idf_vector = feedback.load_matrix(row[0], w2v, idf)

        tmp_rec, tmp_sco = [], []
        for n in range(len(q_matrix)):
            q_sim = similarity.sim_doc_pair(query_matrix, q_matrix[n], query_idf_vector, q_idf_vector[n])
            if q_sim > 0.9:
                if answers[n] not in tmp_rec:
                    tmp_rec.append(answers[n])
                    tmp_sco.append(q_sim)
                else:
                    tmp_sco[tmp_rec.index(answers[n])] += q_sim
        rec_api.append(tmp_rec)
        rec_score.append(tmp_sco)
    print(len(rec_score))
    print(len(rec_api))
    return rec_score, rec_api


get_fr_cal_sim()
