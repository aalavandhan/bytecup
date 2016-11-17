import numpy as np
import pandas as pd
import sys
from os import listdir
from os.path import isfile, join

from util.eval import generate_ndcg_scores
from util.dict_reader import DictReader
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import linear_model

import gc

TRAIN = sys.argv[1]
TEST  = sys.argv[2]
OP    = sys.argv[3]


question_info = pd.read_csv("data/question_info.txt", sep="\t", header=None, names=[
  "question_id", "tag", "word_id", "char_id", "upvotes", "answers", "top_answers"
])
user_info = pd.read_csv("data/user_info.txt", sep="\t", header=None, names=[
  "user_id", "expert_tags", "word_id", "char_id"
])
train = pd.read_csv(TRAIN, sep=",")
test  = pd.read_csv(TEST, sep=",")

vctorizer = CountVectorizer(lambda s: s.split('/'))
qWords = pd.DataFrame( vctorizer.fit_transform(question_info['word_id']).todense() )
qWords.columns = map(lambda i: "qw" + str(i), range(len(qWords.columns)))
qWords = qWords.astype(np.bool)

qChars = pd.DataFrame( vctorizer.fit_transform(question_info['char_id']).todense() )
qChars.columns = map(lambda i: "qc" + str(i), range(len(qChars.columns)))
qChars = qChars.astype(np.bool)

uWords = pd.DataFrame( vctorizer.fit_transform(user_info['word_id']).todense() )
uWords.columns = map(lambda i: "uw" + str(i), range(len(uWords.columns)))
uWords = uWords.astype(np.bool)

uChars = pd.DataFrame( vctorizer.fit_transform(user_info['char_id']).todense() )
uChars.columns = map(lambda i: "uc" + str(i), range(len(uChars.columns)))
uChars = uChars.astype(np.bool)

uTags  = pd.DataFrame( vctorizer.fit_transform(user_info['expert_tags']).todense() )
uTags.columns = map(lambda i: "ut" + str(i), range(len(uTags.columns)))
uTags = uTags.astype(np.bool)

QFEATURES = [ "tag", "upvotes", "answers", "top_answers" ] + qWords.columns.tolist()
UFEATURES = uWords.columns.tolist() + uTags.columns.tolist()

FEATURES = QFEATURES + UFEATURES
TARGET = 'answered'

detailed_question_info = pd.concat([question_info, qWords], axis=1, join_axes=[question_info.index])
detailed_user_info     = pd.concat([user_info, uWords, uTags], axis=1, join_axes=[user_info.index])

TRAIN_DATA = train.merge(detailed_question_info, left_on='question_id', right_on='question_id', how='inner')
TRAIN_DATA = TRAIN_DATA.merge(detailed_user_info, left_on='user_id', right_on='user_id', how='inner')

TEST_DATA = test.merge(detailed_question_info, left_on='question_id', right_on='question_id', how='inner')
TEST_DATA = TEST_DATA.merge(detailed_user_info, left_on='user_id', right_on='user_id', how='inner')

print "DATA LOADED"

model = linear_model.LogisticRegression(
    penalty='l2',
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver='liblinear',
    max_iter=1000,
    multi_class='ovr',
    verbose=1,
    warm_start=False,
    n_jobs=1
)

TRAIN_TARGET = TRAIN_DATA[TARGET].copy()
TRAIN_DATA = TRAIN_DATA[FEATURES].as_matrix()

model.fit(TRAIN_DATA, TRAIN_TARGET)
print "TRAINING DONE"

TEST_DATA = TEST_DATA[FEATURES].as_matrix()
predictions = model.predict_proba(TEST_DATA)
print "PREDICTION DONE"

res = TEST_DATA[['question_id', 'user_id']].copy()
res['answered'] = predictions[:,1]
res.to_csv(OP, sep=",", index=None)
