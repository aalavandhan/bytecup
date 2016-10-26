import numpy as np
import pandas as pd
import scipy
from sklearn import linear_model
from scipy.stats import pearsonr

from recommenders.user_cf import UserCf
from recommenders.item_cf import ItemCf

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


recommender = UserCf(user_info, question_info, train_info, user_index, question_index,NUMBER_OF_USERS, NUMBER_OF_QUESTIONS)
recommender.hyper_parameters(5, -0.01)
recommender.preprocess()

recommendations = test_info.apply(lambda row: recommender.recommend(row['question_id'], row['user_id']), axis=1)

writeRecommendationsToFile(recommendations, test_info, TEST_PATH)
