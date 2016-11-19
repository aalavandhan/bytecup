import numpy as np
import graphlab
import scipy
from scipy import sparse
from user_cf_glab import UserCfGLab

class ItemCfGLab(UserCfGLab):
  def preprocess(self):
    self.base_preprocess()
    trainFrame = graphlab.SFrame({
      'user_id': self.train_info['user_id'],
      'item_id': self.train_info['question_id'],
      'rating' : self.train_info['answered'],
    })
    self.recommender = graphlab.item_similarity_recommender.create(trainFrame,
      target='rating',
      similarity_type=self.similarity)
    testFrame = graphlab.SFrame({
      'user_id': self.test_info['user_id'],
      'item_id': self.test_info['question_id'],
    })
    self.predictions = self.recommender.predict(testFrame)
    return self


