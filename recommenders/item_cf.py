import numpy as np
from scipy.stats import pearsonr

from user_cf import UserCf

class ItemCf(UserCf):
  def _recommend(self, question, user):
    # active question
    active_question = np.array( self.rMatrix[:, self.question_index[question]].transpose() )[ 0 ]

    # Questions the user has been asked
    questions = self.train_info[self.train_info.user_id == user]
    # Questions the user has answered
    questions = questions[questions.answered == 1]['question_id']

    question_indices = map(lambda q: self.question_index[q], questions)
    question_vectors = np.array( self.rMatrix[:, question_indices].transpose() )

    correlation = map(lambda q: (self.pearsoncorr(active_question, q), q), question_vectors)
    closest_question_vectors = sorted(correlation, key=lambda x: x[0], reverse=True)

    top_k = self.topK(closest_question_vectors)

    return active_question.mean() + self.prediction(top_k, self.user_index[user])
