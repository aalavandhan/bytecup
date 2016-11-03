import numpy as np
from scipy.stats import pearsonr

class BaseRecommender:
  def __init__(self, user_info, question_info, train_info,
                      user_index, question_index,
                      NUMBER_OF_USERS, NUMBER_OF_QUESTIONS):

    self.user_info = user_info.copy()
    self.question_info = question_info.copy()
    self.train_info = train_info.copy()
    self.user_index = user_index.copy()
    self.question_index = question_index.copy()
    self.NUMBER_OF_USERS = NUMBER_OF_USERS
    self.NUMBER_OF_QUESTIONS = NUMBER_OF_QUESTIONS

  def pearsoncorr(self, x,y):
    return pearsonr(x, y)[ 0 ]

  def hyper_parameters(self, K, IGNORED):
    # Hyper parameters
    self.K = K
    self.IGNORED = IGNORED
    return self

  def preprocess(self, leave_one_out=False):
    # Do some preprocessing
    # Setting value for ignored
    self.train_info.ix[self.train_info.answered == 0, 'answered'] = self.IGNORED
    self.leave_one_out = leave_one_out
    return self

  def recommend(self, question, user):
    # Return a value from 0-1
    smoothen = lambda x: max(0, min(1, x))
    r = self._recommend(question,user)
    return smoothen(r)

  def expand(self, df):
    row  = [ ]
    col  = [ ]
    data = [ ]
    for i, r in df.iterrows():
      row.append(self.user_index[r['user_id']])
      col.append(self.question_index[r['question_id']])
      data.append(r['answered'])
    return (row, col, data)



