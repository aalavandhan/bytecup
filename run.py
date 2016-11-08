import numpy as np
import pandas as pd
import scipy
from sklearn import linear_model
from scipy.stats import pearsonr

from recommenders.user_cf import UserCf
from recommenders.item_cf import ItemCf
from recommenders.mf      import MF
from recommenders.mf_imp  import MFImp

from recommenders.user_cf_inf import UserCfInf
from recommenders.item_cf_inf import ItemCfInf

from recommenders.text_similarity import TextSimilarity
from recommenders.tag_rule        import TagRule

from recommenders.io import *

import sys

TRAIN_PATH     = sys.argv[1]
TEST_PATH      = sys.argv[2]
OUTPUT_PATH    = sys.argv[3]

REC_TYPE       = eval(sys.argv[4])

if(len(sys.argv) > 5):
  ARGS         = eval(sys.argv[5])
else:
  ARGS         = "[]"

user_info = pd.read_csv("data/user-features")
question_info = pd.read_csv("data/question-features")

train_info = pd.read_csv(TRAIN_PATH, sep=",")

test_info = pd.read_csv(TEST_PATH, sep=",")

NUMBER_OF_USERS = len(user_info)
NUMBER_OF_QUESTIONS = len(question_info)

user_index = { }
for index, row in user_info.iterrows():
  user_index[ row['user_id'] ] = index

question_index = { }
for index, row in question_info.iterrows():
  question_index[ row['question_id'] ] = index

r = REC_TYPE(user_info, question_info, train_info, user_index, question_index,NUMBER_OF_USERS, NUMBER_OF_QUESTIONS)
r.hyper_parameters(*ARGS)
r.preprocess()

def ensemble_recommender(row):
  q = row['question_id']
  u = row['user_id']
  return r.recommend(q,u)

recommendations = test_info.apply(ensemble_recommender, axis=1)

writeRecommendationsToFile(recommendations, test_info, TEST_PATH, OUTPUT_PATH)
