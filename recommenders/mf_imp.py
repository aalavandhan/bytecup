import numpy as np
import pandas as pd
from mf import MF
from sklearn.neighbors import BallTree as BallTree

class MFImp(MF):
  def impute(self, uqMatrix):
    # Updating Cold starts
    for idx, u in self.cold_start.iterrows():
      dists, ids = self.BT.query(self.warm_start[idx, :], k=self.limit)
      uqMatrix[idx, :] = np.amax(uqMatrix[ids[ 0 ], :], axis=0)
    return uqMatrix


  def hyper_parameters(self, K, lb, IGNORED, range=0.01, ca=1):
    # Hyper parameters
    MF.hyper_parameters(self, K, lb, IGNORED, ca)
    self.range = range
    return self

  def preprocess(self):
    self.base_preprocess()
    self.warm_start = self.user_info[self.user_features].as_matrix()
    self.cold_start = self.user_info[ self.user_info.asked == 0 ]
    self.limit = int( len(self.warm_start) * self.range )
    self.BT = BallTree(self.warm_start, leaf_size=self.limit+1, p=2)


    self.uqMatrix = self.impute(
      np.matrix(self.sparse(self.expand(self.train_info)).toarray())
    )

    self.factorize( self.sparse(self.uqMatrix) )

    return self
