import numpy as np
from scipy.stats import pearsonr

from base import BaseRecommender

class UserCf(BaseRecommender):
  def featurize(self, user):
    v = np.repeat(0, self.NUMBER_OF_QUESTIONS)

    def update_vector(r):
      v[ self.question_index[r['question_id']] ] = r['answered']

    # Questions the user has been asked
    questions = self.train_info[self.train_info.user_id == user]

    # Questions that belong to the current set
    questions = questions[questions.question_id.isin(self.question_info['question_id'])]

    questions.apply(update_vector, axis=1)

    return v

  def _recommend(self, question, user):
    qi = self.question_index[question]

    # active user
    active_user = self.featurize(user)

    # users who've been asked the question
    users = self.train_info[self.train_info.question_id == question]
    # users who've answered this question
    users = users[users.answered == 1]['user_id']
    user_vectors = map(self.featurize, users)

    # top K
    top_k = sorted(user_vectors, key=lambda x: self.pearsoncorr(active_user, x) )[ :self.K ]

    # predicted rating
    weighted_sum = reduce(lambda m, u: m + ((u[qi] - u.mean()) * self.pearsoncorr(active_user, u)), top_k, 0)
    sum_of_weights = reduce(lambda m, u: m + self.pearsoncorr(active_user, u), top_k, 0)

    if sum_of_weights == 0 or np.isnan(weighted_sum) or np.isnan(sum_of_weights):
      recommended = 0
    else:
      recommended = active_user.mean() + weighted_sum / sum_of_weights

    return recommended
