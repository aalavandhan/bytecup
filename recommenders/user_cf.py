import numpy as np
from scipy.sparse import csr_matrix

from base import BaseRecommender
from sklearn.neighbors import NearestNeighbors

class UserCf(BaseRecommender):
  def _recommend(self, question, user, index):
    # active user
    active_user = {
      'vector': np.array(self.rMatrix[self.user_index[user], :])[ 0 ],
      'index' : self.user_index[user],
    }

    # users who've been asked the question
    users = self.train_info[self.train_info.question_id == question]
    # users who've answered this question
    users = users[users.answered == 1]['user_id']
    user_indices = map(lambda u: self.user_index[u], users)


    uM = self.rMatrix[user_indices, :]
    aU = self.rMatrix[ self.user_index[user], : ]
    k = min(self.K, len(uM))

    if k == 0:
      return 0

    kNN = NearestNeighbors(n_neighbors=k, algorithm=self.kType, metric=self.distance).fit(uM)
    distances, indics = kNN.kneighbors(aU)
    top_k = uM[indics,:]

    return active_user['vector'].mean() + self.prediction(top_k[ 0 ], distances[ 0 ], self.question_index[question])


  def similarity(self, active, current):
    return self.pearsoncorr(active['vector'], current['vector'])

  def preprocess(self):
    BaseRecommender.preprocess(self)
    V = csr_matrix(self.expand(self.train_info))
    self.rMatrix = V.todense()

    return self

  def prediction(self, top_k, distances, index):
    idx = range(len(top_k))
    common = np.array( top_k[:, index] )[ 0 ] - np.array( top_k[:, ].mean(axis=1) )[ 0 ]
    weighted_sum = (common * distances).sum()
    sum_of_weights = distances.sum()


    # Handling this error
    if sum_of_weights == 0:
      return 0

    return weighted_sum / sum_of_weights
