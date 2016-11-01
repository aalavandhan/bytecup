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


ds1 = question_info[ question_info.answerability <= 0.2  ]
ds2 = question_info[ question_info.answerability > 0.2   ]

# r1 = MF(user_info, ds1, train_info, user_index, question_index,NUMBER_OF_USERS, len(ds1))
# r1.hyper_parameters(10, 0, -0.01)
# r1.preprocess()

# r2 = MF(user_info, ds2, train_info, user_index, question_index,NUMBER_OF_USERS, len(ds2))
# r2.hyper_parameters(10, 0, -0.01)
# r2.preprocess()

# r1 = UserCf(user_info, ds1, train_info, user_index, question_index,NUMBER_OF_USERS, NUMBER_OF_QUESTIONS)
# r1.hyper_parameters(5, -0.01)
# r1.preprocess()

# r2 = UserCf(user_info, ds2, train_info, user_index, question_index,NUMBER_OF_USERS, NUMBER_OF_QUESTIONS)
# r2.hyper_parameters(5, -0.01)
# r2.preprocess()

r = ItemCf(user_info, question_info, train_info, user_index, question_index,NUMBER_OF_USERS, NUMBER_OF_QUESTIONS)
r.hyper_parameters(5, -0.01)
r.preprocess()

def ensemble_recommender(row):
  q = row['question_id']
  u = row['user_id']
  return r.recommend(q,u)

recommendations = test_info.apply(ensemble_recommender, axis=1)

writeRecommendationsToFile(recommendations, test_info, TEST_PATH)
