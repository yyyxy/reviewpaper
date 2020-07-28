from preprocess import similarity
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import SnowballStemmer


def load_matrix(query, w2v, idf):
    query_words = WordPunctTokenizer().tokenize(query.lower())
    if query_words[-1] == '?':
        query_words = query_words[:-1]
    query_words = [SnowballStemmer('english').stem(word) for word in query_words]

    query_matrix = similarity.init_doc_matrix(query_words, w2v)
    query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)
    return query_matrix, query_idf_vector


def get_sim_query(train, test, w2v, idf):
    sim = 0
    for i in range(len(train)):
        train_matrix, train_idf = load_matrix(train, w2v, idf)
        test_matrix, test_idf = load_matrix(test, w2v, idf)
        sim = similarity.sim_doc_pair(train_matrix, test_matrix, train_idf, test_idf)
    return sim


def get_feedback_api(query, answer, query_matrix, query_idf_vector, w2v, idf):
    line = 0
    feeds = []
    for row in answer:
        if line > 0:
            question_matrix, question_idf_vector = load_matrix(query[answer.index(row)], w2v, idf)
            sim = similarity.sim_doc_pair(query_matrix, question_matrix, query_idf_vector, question_idf_vector)
            # 若query与反馈的问题相似，则将反馈问题的api信息加入
            if sim > 0.64:
                for n in range(len(row)):
                    feed = [query[answer.index(row)], row[n], sim]
                    feeds.append(feed)
        line += 1
    feeds = sorted(feeds, key=lambda item: item[2], reverse=True)
    # print('feeds', feeds)
    while len(feeds) < 5:
        feeds.append([0, 0, 0])
    feed_sim = []
    for inf in feeds:
        if len(feed_sim) < 5:
            feed_sim.append(inf[2])
    # print('feed_sim', feed_sim)
    return feeds, feed_sim


def get_feedback_inf(test, question, answer, rec_api_test, w2v, idf):
    feedback_inf = []
    for query in test:
        query_matrix, query_idf_vector = load_matrix(query, w2v, idf)
        feedbacks_inf, feed_sim = get_feedback_api(question, answer, query_matrix, query_idf_vector, w2v, idf)
        for api in rec_api_test[10*test.index(query):10*test.index(query)+10]:
            feed_label = []
            label, i = 0, 0
            for feed in feedbacks_inf[0:5]:
                if feed[1] == api:
                    feed_label.append(feed[2])
                    label += 5-i
                # else:
                #     feed_label.append(0)
                i += 1
            while len(feed_label) < 5:
                feed_label.append(0)
            feed_label.append(round(label/15, 2))
            # # print('feed_label', feed_label)
            # # print(round(label/15, 2))
            # if feedbacks_inf[0:5] != [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]:
            #     feed_label.append(1)
            # else:
            #     feed_label.append(0)
            # # print(feedbacks_inf[0:5])
            feedback_inf.append(feed_label)
    return feedback_inf
