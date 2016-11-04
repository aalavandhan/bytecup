import numpy as np
from scipy.sparse import csr_matrix

from item_cf import ItemCf

class ItemCfInf(ItemCf):
  def similarity(self, active, current):
    inf = self.question_info.ix[current['index']]['ansFreq']
    return self.pearsoncorr(active['vector'], current['vector']) * inf

  def preprocess(self):
    ItemCf.preprocess(self)

    answered = self.train_info[ self.train_info.answered == 1 ].groupby('question_id').count()['answered']
    self.question_info = self.question_info.join(answered, on="question_id", how="left" )
    self.question_info['answered'] =  self.question_info['answered'].fillna(0)
    self.question_info['ansFreq'] = self.NUMBER_OF_QUESTIONS / ( self.question_info['answered'] + 1 )
    self.question_info['ansFreq'] = np.log( self.question_info['ansFreq'] )

    return self
