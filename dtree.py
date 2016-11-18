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
