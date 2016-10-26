import numpy as np
from scipy.stats import pearsonr

from user_cf import UserCf

class ItemCf(UserCf):
  def featurize(self, question):
    v = np.repeat(0, self.NUMBER_OF_USERS)

    def update_vector(r):
      v[ self.user_index[r['user_id']] ] = r['answered']

    # users who've been asked the question
    users = self.train_info[self.train_info.question_id == question]
    # users who've answered this question
    users[users.answered == 1].apply(update_vector, axis=1)

    return v

  def _recommend(self, question, user):
    ui = self.user_index[user]

    # active question
    active_question = self.featurize(question)

    # Questions the user has been asked
    questions = self.train_info[self.train_info.user_id == user]
    # Questions the user has answered
    questions = questions[questions.answered == 1]['question_id']
    question_vectors = map(self.featurize, questions)

    # top K
    top_k = sorted(question_vectors, key=lambda x: self.pearsoncorr(active_question, x) )[ :self.K ]

    # predicted rating
    weighted_sum = reduce(lambda m, u: m + ((u[ui] - u.mean()) * self.pearsoncorr(active_question, u)), top_k, 0)
    sum_of_weights = reduce(lambda m, u: m + self.pearsoncorr(active_question, u), top_k, 0)

    if sum_of_weights == 0 or np.isnan(weighted_sum) or np.isnan(sum_of_weights):
      recommended = 0
    else:
      recommended = active_question.mean() + weighted_sum / sum_of_weights

    return recommended
