import numpy as np
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix

from base import BaseRecommender

class UserCf(BaseRecommender):
  def _recommend(self, question, user):
    # active user
    active_user = np.array(self.rMatrix[self.user_index[user], :])[ 0 ]

    # users who've been asked the question
    users = self.train_info[self.train_info.question_id == question]
    # users who've answered this question
    users = users[users.answered == 1]['user_id']

    user_indices = map(lambda u: self.user_index[u], users)
    user_vectors = np.array(self.rMatrix[user_indices, :])

    correlation = map(lambda u: (self.pearsoncorr(active_user, u), u), user_vectors)
    closest_user_vectors = sorted(correlation, key=lambda x: x[0], reverse=True)

    top_k = self.topK(closest_user_vectors)

    return active_user.mean() + self.prediction(top_k, self.question_index[question])

  def preprocess(self, leave_one_out=False):
    BaseRecommender.preprocess(self)

    (row, col, data) = self.expand(self.train_info)
    V = csr_matrix((data, (row, col)))
    self.rMatrix = V.todense()

    self.leave_one_out = leave_one_out

    return self

  def prediction(self, top_k, index):
    weighted_sum = reduce(lambda m, u: m + ((u[1][index] - u[1].mean()) * u[0]), top_k, 0)
    sum_of_weights = reduce(lambda m, u: m + u[0], top_k, 0)

    if sum_of_weights == 0:
      return 0

    return weighted_sum / sum_of_weights


  def topK(self, closest):
    if not self.leave_one_out:
      top_k = closest[ :self.K ]
    else:
      top_k = closest[ 1:self.K+1 ]

    return top_k
