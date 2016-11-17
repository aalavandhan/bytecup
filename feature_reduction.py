import numpy as np
import pandas as pd
import sys
from os import listdir
from os.path import isfile, join

from sklearn.feature_extraction.text import CountVectorizer

import libpmf.libpmf as libpmf

TYPE = sys.argv[1]
K    = int(sys.argv[2])
lb   = float(sys.argv[3])

question_info = pd.read_csv("data/question_info.txt", sep="\t", header=None, names=[
  "question_id", "tag", "word_id", "char_id", "upvotes", "answers", "top_answers"
])
user_info = pd.read_csv("data/user_info.txt", sep="\t", header=None, names=[
  "user_id", "expert_tags", "word_id", "char_id"
])

vctorizer = CountVectorizer(lambda s: s.split('/'))

if TYPE == "qW":
  data = vctorizer.fit_transform(question_info['word_id'])

if TYPE == "qC":
  data = vctorizer.fit_transform(question_info['char_id'])

if TYPE == "uW":
  data = vctorizer.fit_transform(user_info['word_id'])

if TYPE == "uC":
  data = vctorizer.fit_transform(user_info['char_id'])

if TYPE == "uT":
  data  = vctorizer.fit_transform(user_info['expert_tags'])

model = libpmf.train(data, '-k {0} -l {1} -t 5000'.format(K, lb))

dMAtrix = data.todense()
factorized = np.dot( model['W'], model['H'].transpose() )

ERROR = np.square(dMAtrix - factorized).mean()

print "{0},{1},{2},{3}\n".format(TYPE, K, lb, ERROR)
