import numpy as np
from scipy.stats import pearsonr

class BaseRecommender:
  def __init__(self, user_info, question_info, train_info,
                      user_index, question_index,
                      NUMBER_OF_USERS, NUMBER_OF_QUESTIONS):

    self.user_info = user_info.copy()
    self.question_info = question_info.copy()
    self.train_info = train_info.copy()
    self.user_index = user_index.copy()
    self.question_index = question_index.copy()
    self.NUMBER_OF_USERS = NUMBER_OF_USERS
    self.NUMBER_OF_QUESTIONS = NUMBER_OF_QUESTIONS


  def pearsoncorr(self, x,y):
    return pearsonr(x, y)[ 0 ]

  def hyper_parameters(self, K, IGNORED):
    # Set Hyper parameters
    return self

  def preprocess(self):
    # Do some recommender specific preprocessing
    return self

  def recommend(self, question, user):
    # Return a value from 0-1
    return max(0, min(1, self._recommend(question,user)))
