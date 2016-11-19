import numpy as np
import graphlab
import scipy
from scipy import sparse
from base import BaseRecommender

class MFGLab(BaseRecommender):
  def _recommend(self, question, user, index):
    return self.predictions[index]

  # 0 => ranking_factorization_recommender
  # 1 => factorization_recommender
  # 2 => popularity_recommender
  def hyper_parameters(self, type=0, lb=0.1, IGNORED=0, ca=1):
    # Hyper parameters
    self.type = type
    self.lb = lb
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

    if self.type == 0:
      self.recommender = graphlab.ranking_factorization_recommender.create(trainFrame,
        target='rating',
        regularization=self.lb,
        unobserved_rating_value=0.25)
    elif self.type == 1:
      self.recommender = graphlab.factorization_recommender.create(trainFrame,
        target='rating',
        regularization=self.lb)
    elif self.type == 2:
      self.recommender = graphlab.popularity_recommender.create(trainFrame,
        target='rating')

    testFrame = graphlab.SFrame({
      'item_id': self.test_info['question_id'],
      'user_id': self.test_info['user_id'],
    })
    self.predictions = self.recommender.predict(testFrame)
    return self


