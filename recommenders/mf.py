import numpy as np
import libpmf.libpmf as libpmf
import scipy
from scipy import sparse
from base import BaseRecommender

class MF(BaseRecommender):
  def _recommend(self, question, user):
    return self.factorized[ self.user_index[user] ][ self.question_index[question] ]


  def hyper_parameters(self, K, lb, IGNORED):
    # Hyper parameters
    self.K = K
    self.lb = lb
    self.IGNORED = IGNORED
    return self

  def expand(self, df):
    row  = [ ]
    col  = [ ]
    data = [ ]
    for i, r in df.iterrows():
      row.append(self.user_index[r['user_id']])
      col.append(self.question_index[r['question_id']])
      data.append(r['answered'])
    return (row, col, data)


  def preprocess(self):
    BaseRecommender.preprocess(self)

    # Setting value for ignored
    (row, col, data) = self.expand(self.train_info)
    V = scipy.sparse.csr_matrix((data, (row, col)))
    model = libpmf.train(V, '-k {0} -l {1} -t {0}'.format(self.K, self.lb))

    self.factorized = np.dot( model['W'], model['H'].transpose() )
    return self


