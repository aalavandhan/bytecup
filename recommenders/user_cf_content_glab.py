import numpy as np
import graphlab
import scipy
from scipy import sparse
from user_cf_glab import UserCfGLab

class UserCfContentGLab(UserCfGLab):
  def preprocess(self):
    self.base_preprocess()
    trainFrame = graphlab.SFrame({
      'item_id': self.train_info['user_id'],
      'user_id': self.train_info['question_id'],
      'rating' : self.train_info['answered'],
    })
    qFeatures = reduce(lambda m,f: m.update({ f: self.question_info[f] }) or m, self.question_features, {
      "user_id": self.question_info.index,
    })
    user_info = graphlab.SFrame(qFeatures)
    uFeatures = reduce(lambda m,f: m.update({ f: self.user_info[f] }) or m, self.user_features, {
      "item_id": self.user_info.index,
    })
    item_info = graphlab.SFrame(uFeatures)

    self.recommender = graphlab.item_similarity_recommender.create(trainFrame,
      user_data=user_info,
      item_data=item_info,
      target='rating',
      similarity_type=self.similarity)

    testFrame = graphlab.SFrame({
      'item_id': self.test_info['user_id'],
      'user_id': self.test_info['question_id'],
    })
    self.predictions = self.recommender.predict(testFrame)
    return self


