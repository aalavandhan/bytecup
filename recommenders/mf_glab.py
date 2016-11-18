import numpy as np
import graphlab
import libpmf.libpmf as libpmf
import scipy
from scipy import sparse
from base import BaseRecommender

class MFGLab(BaseRecommender):
  def _recommend(self, question, user):
    return self.recommender.predict({ 'user_id' : self.user_index[user], 'item_id': self.question_index[question] })[ 0 ]

  def hyper_parameters(self, K, IGNORED, ca=1):
    # Hyper parameters
    self.K = K
    self.IGNORED = IGNORED
    self.ca = ca
    return self

  def base_preprocess(self):
    BaseRecommender.preprocess(self)

  def preprocess(self):
    self.base_preprocess()
    (data, (row, col)) = self.expand(self.train_info)

    sf = graphlab.SFrame({
      'user_id': row,
      'item_id': col,
      'rating' : data,
    })
    self.recommender = graphlab.factorization_recommender.create(sf, target='rating')
    return self


