import numpy as np
from mf import MF

class MFImp(MF):
  def impute(self, train_data):
    td = train_data.copy()

    cold_start = self.user_info[ self.user_info.asked == 0 ]
    warm_start = self.user_info[ self.user_info.asked != 0 ]

    import pdb
    pdb.set_trace()

    return td


  def hyper_parameters(self, K, lb, IGNORED, range=0.01, ca=1):
    # Hyper parameters
    self.K = K
    self.lb = lb
    self.IGNORED = IGNORED
    self.range = range
    self.ca = ca
    return self

  def preprocess(self):
    # Generate user similarity matrix
    self.user_similarity = np.identity(self.NUMBER_OF_USERS)
    for i in range(self.NUMBER_OF_USERS):
      self.user_similarity[i,:] = self.user_info.apply(lambda u: self.pearsoncorr(u[self.user_features], self.user_info.ix[i][self.user_features]), axis=1)


    self.train_info = self.impute(self.train_info)
    MF.preprocess(self)
    return self
