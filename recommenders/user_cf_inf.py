import numpy as np
from scipy.sparse import csr_matrix

from user_cf import UserCf

class UserCfInf(UserCf):
  def similarity(self, active, current):
    inf = self.user_info.ix[current['index']]['ansFreq']
    return self.pearsoncorr(active['vector'], current['vector']) * inf

  def preprocess(self):
    UserCf.preprocess(self)

    answered = self.train_info[ self.train_info.answered == 1 ].groupby('user_id').count()['answered']
    self.user_info = self.user_info.join(answered, on="user_id", how="left" )
    self.user_info['answered'] =  self.user_info['answered'].fillna(0)
    self.user_info['ansFreq'] = self.NUMBER_OF_QUESTIONS / ( self.user_info['answered'] + 1 )
    self.user_info['ansFreq'] = np.log( self.user_info['ansFreq'] )

    return self
