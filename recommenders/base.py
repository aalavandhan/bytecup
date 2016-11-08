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

    answered = self.train_info[ self.train_info.answered == 1 ].groupby('user_id').count()['answered']
    asked = self.train_info.groupby('user_id').count()['answered'].rename('asked')
    self.user_info = self.user_info.join(answered, on="user_id", how="left" ).join(asked, on="user_id", how="left" )
    self.user_info['answered'] =  self.user_info['answered'].fillna(0)
    self.user_info['asked']    =  self.user_info['asked'].fillna(0)


    answered = self.train_info[ self.train_info.answered == 1 ].groupby('question_id').count()['answered']
    asked = self.train_info.groupby('question_id').count()['answered'].rename('asked')
    self.question_info = self.question_info.join(answered, on="question_id", how="left" ).join(asked, on="question_id", how="left")
    self.question_info['answered'] =  self.question_info['answered'].fillna(0)
    self.question_info['asked']    =  self.question_info['asked'].fillna(0)

    self.user_features = [
      "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10",
      "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10",
      "t1", "t2", "t3", "t4", "t5",
    ]

  def pearsoncorr(self, x,y):
    c = pearsonr(x, y)[ 0 ]
    return c if not np.isnan(c) else 0

  def hyper_parameters(self, K=7, IGNORED=0.0001, ca=1):
    # Hyper parameters
    self.K = K
    self.IGNORED = IGNORED
    self.ca = ca
    return self

  def preprocess(self, leave_one_out=False):
    # Do some preprocessing
    # Setting value for ignored
    if hasattr(self, 'IGNORED'):
      self.train_info.ix[self.train_info.answered == 0, 'answered'] = self.IGNORED

    self.leave_one_out = leave_one_out
    return self

  def recommend(self, question, user):
    # Return a value from 0-1
    smoothen = lambda x: max(0, min(1, x))
    r = self._recommend(question,user)

    if hasattr(self, 'ca'):
      pw = self.ca
    else:
      pw = 1

    return smoothen(r) ** pw

  def expand(self, df):
    row  = [ ]
    col  = [ ]
    data = [ ]
    for i, r in df.iterrows():
      row.append(self.user_index[r['user_id']])
      col.append(self.question_index[r['question_id']])
      data.append(r['answered'])
    return (data, (row, col))



