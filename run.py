import numpy as np
import pandas as pd
import scipy
from sklearn import linear_model
from scipy.stats import pearsonr

import sys

UNASKED = 0
IGNORED = -0.001

def pearsoncorr(x,y):
  return pearsonr(x, y)[ 0 ]

def featurize(user):
  v = np.repeat(0, NUMBER_OF_QUESTIONS)

  def update_vector(r):
    v[ question_index[r['question_id']] ] = r['answered']

  # Questions the user has answered
  train_info[train_info.user_id == user].apply(update_vector, axis=1)

  return v

# User-User CF
def recommend(question, user, K=3):
  qi = question_index[ question]

  # active user
  active_user = featurize(user)

  # users who've be asked this question
  users = train_info[train_info.question_id == question]
  # users who've answered this question
  users = users[users.answered == 1]['user_id']
  user_vectors = map(featurize, users)

  # top K
  top_k = sorted(user_vectors, key=lambda x: pearsoncorr(active_user, x) )[ :K ]

  # predicted rating
  weighted_sum = reduce(lambda m, u: m + ((u[qi] - u.mean()) * pearsoncorr(active_user, u)), top_k, 0)
  sum_of_weights = reduce(lambda m, u: m + pearsoncorr(active_user, u), top_k, 0)

  if sum_of_weights == 0 or np.isnan(weighted_sum) or np.isnan(sum_of_weights):
    recommended = 0
  else:
    recommended = active_user.mean() + weighted_sum / sum_of_weights

  return min(1, recommended)


ip = sys.argv[1]

user_info = pd.read_csv("data/user-features")
question_info = pd.read_csv("data/question-features")
train_info = pd.read_csv("data/invited_info_train.txt", sep="\t", header=None, names=[
  "question_id", "user_id", "answered"
])

train_info.ix[train_info.answered == 0, 'answered'] = -1

test_info = pd.read_csv(ip, sep=",", header=None, names=[
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

# Setting value for ignored
train_info.ix[train_info.answered == 0, 'answered'] = IGNORED

recommendations = test_info.apply(lambda row: recommend(row['question_id'], row['user_id']), axis=1)

res = test_info[['question_id', 'user_id']].copy()
res['prediction'] = recommendations
res.columns = ['qid','uid','label']
res.to_csv(ip + '.results', sep=",", index=None, header=None)
