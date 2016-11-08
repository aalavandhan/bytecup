import numpy as np
import pandas as pd

from base import BaseRecommender

class TagRule(BaseRecommender):
  def _recommend(self, question, user):
    ui = self.user_index[user]
    qi = self.question_index[question]

    past_tags   = set( self.rule[ self.question_info.ix[qi]['tag'] ] )
    user_tags   = set( self.uTags[self.uTags.user_id == ui]['tag_id'] )

    common = past_tags.intersection( user_tags )
    union  = past_tags.union( user_tags )

    return float(len(common)) / len(past_tags)

  def preprocess(self):
    BaseRecommender.preprocess(self)

    self.uTags = pd.DataFrame.from_csv("data/user-tag", header=None, index_col=None)
    self.uTags.columns=['user_id', 'tag_id']


    # Learn rules dynamically based on tresh
    self.rule = {
      0: [27, 28, 29, 30, 71, 72],
      1: [31],
      2: [2, 3, 32, 33, 39, 41, 65, 66, 74, 75, 103],
      3: [18, 19, 20],
      4: [75],
      5: [0, 1, 23, 31, 35, 36, 37, 39, 44, 47, 57, 58, 59, 68, 75],
      6: [0,7,8,18,19,20,23,25,32,33,38,39,42,43,51,52,53,54,69,76,77,85,95],
      7: [23, 32],
      8: [0, 1, 48, 49, 50, 61, 64],
      9: [2, 3, 4, 5, 6, 24, 65, 66, 67],
      10: [23, 47],
      11: [33, 51, 52, 53, 54],
      12: [33, 34, 40, 60, 100],
      13: [8, 18, 20, 23, 25, 33, 38, 42, 43, 51, 52, 53, 54, 69, 88],
      14: [7, 63, 77, 94],
      15: [21, 22, 33, 51, 53, 79, 80],
      16: [35, 36, 37, 44],
      17: [104, 105, 106, 111, 115],
      18: [23, 32, 57, 62, 104, 105, 106, 107, 111, 115],
      19: [0, 1, 13, 14, 15, 78, 86, 87]
    }
