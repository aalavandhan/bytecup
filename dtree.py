import numpy as np
import pandas as pd
import sys
from os import listdir
from os.path import isfile, join

from util.eval import generate_ndcg_scores
from util.dict_reader import DictReader
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer

VALIDATION_SET = sys.argv[1]
TEST_SET       = sys.argv[2]

VALIDATE       = sys.argv[3]
TEST           = sys.argv[4]

RESULT         = sys.argv[5]


validation_recommenders = map(lambda l: join(VALIDATE, l), listdir(VALIDATE))
test_recommenders  = map(lambda l: join(TEST, l), listdir(TEST))

numberOfRecom = len(validation_recommenders)

results = { }
scores  = { }
mScore  = { }
question = { }
recommendation = { }

for r in range(numberOfRecom):
  results[r] = generate_ndcg_scores(VALIDATION_SET, validation_recommenders[r])
  scores[r]  = map(lambda s: s[0], results[r])
  mScore[r] = np.array(scores[r]).mean()

  recommendation[r] = DictReader(test_recommenders  [r]).load()

  for s in results[r]:
    if s[1] not in question:
      question[s[1]] = { }

    question[s[1]][r] = s[0]


# Build question features
question_info = pd.read_csv("data/question_info.txt", sep="\t", header=None, names=[
    "question_id", "tag", "word_id", "char_id", "upvotes", "answers", "top_answers"
])
train_info = pd.read_csv(VALIDATION_SET, sep=",")
test_info   = pd.read_csv(TEST_SET, sep=",")

# FEATURES = [
#   "tag", "upvotes", "answers", "top_answers",'answerability',
#   # 'nTag', 'votability', 'ease', 'popularity', 'answerability',
#   "wa1", "wa2", "wa3", "wa4", "wa5", "wa6", "wa7", "wa8", "wa9", "wa10",
#   "wb1", "wb2", "wb3", "wb4", "wb5", "wb6", "wb7", "wb8", "wb9", "wb10",
#   "wc1", "wc2", "wc3", "wc4", "wc5", "wc6", "wc7", "wc8", "wc9", "wc10",
#   "wd1", "wd2", "wd3", "wd4", "wd5", "wd6", "wd7", "wd8", "wd9", "wd10",
#   "we1", "we2", "we3", "we4", "we5", "we6", "we7", "we8", "we9", "we10",
#   "wf1",
#   "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10",
#   "cb1", "cb2", "cb3", "cb4", "cb5", "cb6", "cb7", "cb8", "cb9", "cb10",
#   "cc1", "cc2", "cc3", "cc4", "cc5", "cc6", "cc7", "cc8", "cc9", "cc10",
#   "cd1", "cd2", "cd3", "cd4", "cd5", "cd6", "cd7", "cd8", "cd9", "cd10",
#   "ce1", "ce2", "ce3", "ce4", "ce5", "ce6", "ce7", "ce8", "ce9", "ce10",
#   "cf1",
# ]




vctorizer = CountVectorizer(lambda s: s.split('/'))
qWords = pd.DataFrame( vctorizer.fit_transform(question_info['word_id']).todense() )
qWords.columns = map(lambda i: "w" + str(i), range(len(qWords.columns)))
qChars = pd.DataFrame( vctorizer.fit_transform(question_info['char_id']).todense() )
qChars.columns = map(lambda i: "c" + str(i), range(len(qChars.columns)))

question_info = pd.concat([question_info, qWords, qChars], axis=1, join_axes=[question_info.index])

FEATURES = [ "tag", "upvotes", "answers", "top_answers" ] + qWords.columns.tolist() + qChars.columns.tolist()

def assign_recommender(r):
  s = sorted(range(numberOfRecom), key=lambda rec: (question[ r['question_id'] ][ rec ], mScore[ rec ]), reverse=True)
  return s[0]

validation_data = question_info[question_info.question_id.isin(train_info['question_id'])].copy()
validation_data['classification'] = validation_data.apply(lambda r: assign_recommender(r), axis=1)
test_data = question_info[question_info.question_id.isin(test_info['question_id'])].copy()


# Train
c = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None,
  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
  max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07,
  bootstrap=True, oob_score=False, n_jobs=1, random_state=None,
  verbose=0, warm_start=False, class_weight=None)


c.fit(validation_data[FEATURES], validation_data['classification'])
test_data['recommendation'] = c.predict(test_data[FEATURES])

# Return recommendation
def recommend(r):
  rec = test_data[ test_data.question_id == r['question_id'] ]['recommendation'].tolist()[ 0 ]
  return float( recommendation[rec][r['question_id']][r['user_id']] )

test_info['recommendation'] = test_info.apply(recommend, axis=1)
test_info[['question_id', 'user_id', 'recommendation']].to_csv(RESULT, sep=",", index=None)
