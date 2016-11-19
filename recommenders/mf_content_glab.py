import numpy as np
import graphlab
import scipy
from scipy import sparse
from mf_glab import MFGLab

class MFContentGLab(MFGLab):
  def preprocess(self):
    self.base_preprocess()
    trainFrame = graphlab.SFrame({
      'user_id': self.train_info['user_id'].tolist(),
      'item_id': self.train_info['question_id'].tolist(),
      'rating' : self.train_info['answered'].tolist(),
    })
    qFeatures = reduce(lambda m,f: m.update({ f: self.question_info[f] }) or m, self.question_features, {
      "item_id": self.question_info.index,
    })
    item_info = graphlab.SFrame(qFeatures)
    uFeatures = reduce(lambda m,f: m.update({ f: self.user_info[f] }) or m, self.user_features, {
      "user_id": self.user_info.index,
    })
    user_info = graphlab.SFrame(uFeatures)

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
      'user_id': self.test_info['user_id'].tolist(),
      'item_id': self.test_info['question_id'].tolist(),
    })
    self.predictions = self.recommender.predict(testFrame)

    return self


