import numpy as np
import graphlab
import scipy
from scipy import sparse
from base import BaseRecommender

class MFContentGLab(BaseRecommender):
  def _recommend(self, question, user, index):
    return self.predictions[index]

  def hyper_parameters(self, IGNORED, ca=1):
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

    self.recommender = graphlab.ranking_factorization_recommender.create(trainFrame,
      target='rating',
      similarity_type='pearson')
    testFrame = graphlab.SFrame({
      'user_id': self.test_info['user_id'].tolist(),
      'item_id': self.test_info['question_id'].tolist(),
    })
    self.predictions = self.recommender.predict(testFrame)

    return self


