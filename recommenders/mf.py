import numpy as np
import libpmf.libpmf as libpmf
import scipy
from scipy import sparse
from base import BaseRecommender

class MF(BaseRecommender):
  def _recommend(self, question, user):
    return self.factorized[ self.user_index[user] ][ self.question_index[question] ]

  def hyper_parameters(self, K, lb, IGNORED, ca=1):
    # Hyper parameters
    self.K = K
    self.lb = lb
    self.IGNORED = IGNORED
    self.ca = ca
    return self

  def base_preprocess(self):
    BaseRecommender.preprocess(self)


  def factorize(self, V):
    model = libpmf.train(V, '-k {0} -l {1} -t 5000'.format(self.K, self.lb))
    self.factorized = np.dot( model['W'], model['H'].transpose() )


  def sparse(self, d):
    return scipy.sparse.csr_matrix(d)

  def preprocess(self):
    self.base_preprocess()
    V = self.sparse(self.expand(self.train_info))
    self.factorize(V)

    return self


