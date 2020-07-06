# recommendation API item
class APIRec:

    def __init__(self, init_id, title, api_description):
        self.init_id = init_id  # api initial ranking position
        # self.resort_id = 0  # api reranking position
        self.title = title
        self.api_description = api_description
        # api feature vector
        self.api_relate_sim = [0 for n in range(2)]
        self.feedback_sim = [0 for n in range(5)]
        self.pred = 0
        self.pred_LTR = 0
        self.pred_AL = 0

    def print_info(self):
        print(self.init_id, self.title, self.api_description, self.api_relate_sim, self.feedback_sim)


# feedback repository data structure
class FeedbackInfo:

    def __init__(self, query, api):
        self.query = query
        self.api = api
        self.query_matrix = []
        self.query_idf_vector = []


# feature vector structure defination
class FeatureVector:

    def __init__(self, label, feature, title):
        self.id = id
        self.feature = feature
        self.title = title
        self.feedback_sim = [0 for n in range(5)]
        self.label = int(label)

    def update_feature(self):
        self.feature[2:]=self.feedback_sim

    def get_title(self):
        return self.title


