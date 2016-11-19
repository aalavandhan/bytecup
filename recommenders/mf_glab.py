import numpy as np
import graphlab
import scipy
from scipy import sparse
from base import BaseRecommender

class MFGLab(BaseRecommender):
  def _recommend(self, question, user, index):
    return self.predictions[index]

  def hyper_parameters(self, IGNORED=0, ca=1):
    # Hyper parameters
    self.IGNORED = IGNORED
    self.ca = ca
    return self

  def base_preprocess(self):
    BaseRecommender.preprocess(self)

  def preprocess(self):
    self.base_preprocess()
    trainFrame = graphlab.SFrame({
      'user_id': self.train_info['user_id'].tolist(),
      'item_id': self.train_info['question_id'].tolist(),
      'rating' : self.train_info['answered'],
    })
    self.recommender = graphlab.ranking_factorization_recommender.create(trainFrame,target='rating')
    testFrame = graphlab.SFrame({
      'item_id': self.test_info['question_id'],
      'user_id': self.test_info['user_id'],
    })
    self.predictions = self.recommender.predict(testFrame)
    return self


