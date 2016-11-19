import numpy as np
import graphlab
import scipy
from scipy import sparse
from mf_glab import MFGLab

class UserCfGLab(MFGLab):
  def hyper_parameters(self, similarity='pearson',IGNORED=0, ca=1):
    # Hyper parameters
    self.similarity = similarity
    self.IGNORED = IGNORED
    self.ca = ca
    return self

  def preprocess(self):
    self.base_preprocess()
    trainFrame = graphlab.SFrame({
      'item_id': self.train_info['user_id'],
      'user_id': self.train_info['question_id'],
      'rating' : self.train_info['answered'],
    })
    self.recommender = graphlab.item_similarity_recommender.create(trainFrame,
      target='rating',
      similarity_type=self.similarity)
    testFrame = graphlab.SFrame({
      'item_id': self.test_info['user_id'],
      'user_id': self.test_info['question_id'],
    })
    self.predictions = self.recommender.predict(testFrame)
    return self


