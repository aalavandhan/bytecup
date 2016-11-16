import numpy as np
import pandas as pd

from base import BaseRecommender

class TextSimilarity(BaseRecommender):
  def _recommend(self, question, user):
    ui = self.user_index[user]
    qi = self.question_index[question]

    s1 = self.qChar[ self.qChar.question_id == qi ]['char_id']
    s2 = self.uChar[ self.uChar.user_id     == ui ]['char_id']

    s3 = self.qWord[ self.qWord.question_id == qi ]['word_id']
    s4 = self.uWord[ self.uWord.user_id     == ui ]['word_id']


    char_sim = self.similarity(set(s1), set(s2))
    word_sim = self.similarity(set(s3), set(s4))

    if self.type == "word":
        return word_sim
    else:
        return char_sim


  def preprocess(self):
    BaseRecommender.preprocess(self)

    self.qWord = pd.DataFrame.from_csv("data/question-word", header=None, index_col=None)
    self.qWord.columns=['question_id', 'word_id']
    self.qChar = pd.DataFrame.from_csv("data/question-char", header=None, index_col=None)
    self.qChar.columns=['question_id', 'char_id']
    self.uWord = pd.DataFrame.from_csv("data/user-word", header=None, index_col=None)
    self.uWord.columns=['user_id', 'word_id']
    self.uChar = pd.DataFrame.from_csv("data/user-char", header=None, index_col=None)
    self.uChar.columns=['user_id', 'char_id']


  def similarity(self, v1, v2):
    u = v1.union(v2)
    i = v1.intersection(v2)
    return float( len(i) ) / len(u)

  def hyper_parameters(self, type='word', ca=1):
    # Hyper parameters
    self.type = type
    self.ca = ca
    return self
