import numpy as np
import pandas as pd
import scipy
from sklearn import linear_model
from scipy.stats import pearsonr

from recommenders.user_cf import UserCf
from recommenders.item_cf import ItemCf
from recommenders.mf      import MF

from recommenders.io import *

import sys

TRAIN_PATH   = sys.argv[1]
TEST_PATH    = sys.argv[2]

K            = int(sys.argv[3])

user_info = pd.read_csv("data/user-features")
question_info = pd.read_csv("data/question-features")

train_info = pd.read_csv(TRAIN_PATH, sep="\t", header=None, names=[
  "question_id", "user_id", "answered"
])

test_info = pd.read_csv(TEST_PATH, sep=",", header=None, names=[
  "question_id", "user_id", "answered"
])

NUMBER_OF_USERS = len(user_info)
NUMBER_OF_QUESTIONS = len(question_info)

user_index = { }
for index, row in user_info.iterrows():
  user_index[ row['user_id'] ] = index

question_index = { }
for index, row in question_info.iterrows():
  question_index[ row['question_id'] ] = index

r = UserCf(user_info, question_info, train_info, user_index, question_index,NUMBER_OF_USERS, NUMBER_OF_QUESTIONS)
r.hyper_parameters(K, 0)
r.preprocess(leave_one_out=True)

def ensemble_recommender(row):
  q = row['question_id']
  u = row['user_id']
  return r.recommend(q,u)

recommendations = test_info.apply(ensemble_recommender, axis=1)

writeRecommendationsToFile(recommendations, test_info, TEST_PATH)
