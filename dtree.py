import numpy as np
import pandas as pd
import sys
from os import listdir
from os.path import isfile, join

from util.eval import generate_ndcg_scores
from util.dict_reader import DictReader
from sklearn.ensemble import RandomForestClassifier

import graphlab

from operator import itemgetter

VALIDATION_SET = sys.argv[1]
TEST_SET       = sys.argv[2]
VALIDATE       = sys.argv[3]
TEST           = sys.argv[4]
RESULT         = sys.argv[5]

validation_recommenders = sorted( map(lambda l: join(VALIDATE, l), listdir(VALIDATE)) )
test_recommenders       = sorted( map(lambda l: join(TEST, l), listdir(TEST)) )

print validation_recommenders
print test_recommenders

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


train_info = pd.read_csv(VALIDATION_SET, sep=",")
test_info   = pd.read_csv(TEST_SET, sep=",")

# Build question features
question_info = pd.read_csv("data/question-features")

answered = train_info[ train_info.answered == 1 ].groupby('question_id').count()['answered']
asked = train_info.groupby('question_id').count()['answered'].rename('asked')
question_info = question_info.join(answered, on="question_id", how="left" ).join(asked, on="question_id", how="left")
question_info['answered'] =  question_info['answered'].fillna(0)
question_info['asked']    =  question_info['asked'].fillna(0)
question_info['asked']    =  question_info['asked'] + question_info['answered']

question_info['answerability'] = ( question_info['top_answers'] / question_info['answers'] )
question_info['answerability'] =  question_info['answerability'].fillna(-1)

FEATURES = list( set(question_info.columns) - set([ "answered", "question_id"]) )

def assign_recommender(r):
  s = sorted(range(numberOfRecom), key=lambda rec: (question[ r['question_id'] ][ rec ], mScore[ rec ]), reverse=True) 
  return s[0]
  srs = question[ r['question_id'] ] 
  ordered = sorted(srs.items(), key=itemgetter(1), reverse=True)
  if ordered[0][1] == 0:
    return s[0]
  best = filter(lambda s: s[1] == ordered[0][1], ordered)
  bestRecommenders = map(lambda b: b[0], best)
  bestScores = np.array( map(lambda b: b[1], best) )
  return np.random.choice(bestRecommenders, p=bestScores/bestScores.sum())

validation_data = question_info[question_info.question_id.isin(train_info['question_id'])].copy()
validation_data['classification'] = validation_data.apply(lambda r: assign_recommender(r), axis=1)
test_data = question_info[question_info.question_id.isin(test_info['question_id'])].copy()


# print "Training .. "
# Train
# c = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None,
#   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
#   max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07,
#   bootstrap=True, oob_score=False, n_jobs=1, random_state=None,
#   verbose=0, warm_start=False, class_weight=None)
# c.fit(validation_data[FEATURES], validation_data['classification'])
# print "Predicting .. "
# test_data['recommendation'] = c.predict(test_data[FEATURES])
print validation_data.groupby(['classification']).count()

model = graphlab.random_forest_classifier.create(graphlab.SFrame(data=validation_data), target='classification', features=FEATURES)
test_data['recommendation'] = model.predict(graphlab.SFrame(data=test_data[FEATURES]))

print test_data.groupby(['recommendation']).count()

# Return recommendation
def recommend(r):
  rec = test_data[ test_data.question_id == r['question_id'] ]['recommendation'].tolist()[ 0 ]
  return float( recommendation[rec][r['question_id']][r['user_id']] )

test_info['answered'] = test_info.apply(recommend, axis=1)
test_info[['question_id', 'user_id', 'answered']].to_csv(RESULT, sep=",", index=None)
