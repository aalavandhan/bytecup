import numpy as np
from scipy.stats import pearsonr

from user_cf import UserCf
from sklearn.neighbors import NearestNeighbors

class ItemCf(UserCf):
  def _recommend(self, question, user):
    # active question
    active_question = {
      'vector': np.array( self.rMatrix[:, self.question_index[question]].transpose() )[ 0 ],
      'index': self.question_index[question],
    }

    # Questions the user has been asked
    questions = self.train_info[self.train_info.user_id == user]
    # Questions the user has answered
    questions = questions[questions.answered == 1]['question_id']
    question_indices = map(lambda q: self.question_index[q], questions)


    uM = self.rMatrix[question_indices, :]
    aU = self.rMatrix[ self.question_index[question], : ]
    k = min(self.K, len(uM))

    if k == 0:
      return 0

    kNN = NearestNeighbors(n_neighbors=k, algorithm=self.kType, metric=self.distance).fit(uM)
    distances, indics = kNN.kneighbors(aU)
    top_k = uM[indics,:]

    return active_question['vector'].mean() + self.prediction(top_k[ 0 ], distances[ 0 ], self.user_index[user])


  def preprocess(self):
    UserCf.preprocess(self)
    self.rMatrix = self.rMatrix.transpose()
    return self
