import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Build question features
question_info = pd.read_csv("data/question_info.txt", sep="\t", header=None, names=[
    "question_id", "tag", "word_id", "char_id", "upvotes", "answers", "top_answers"
])

question_info['ease'] = ( question_info['answers'] - question_info['answers'].min()  ) / question_info['answers'].max()
question_info['popularity'] = ( question_info['top_answers'] - question_info['top_answers'].min() ) / question_info['top_answers'].max()
question_info['votability'] = ( question_info['upvotes'] - question_info['upvotes'].min() ) / question_info['upvotes'].max()
question_info['nTag'] = ( question_info['tag'] - question_info['tag'].min() ) / question_info['tag'].max()
question_info['answerability'] = ( question_info['top_answers'] / question_info['answers'] )
question_info['answerability'] =  question_info['answerability'].fillna(0)


vctorizer = CountVectorizer(lambda s: s.split('/'))
# qWords = pd.DataFrame( vctorizer.fit_transform(question_info['word_id']).todense() )
# qWords.columns = map(lambda i: "w" + str(i), range(len(qWords.columns)))
qChars = pd.DataFrame( vctorizer.fit_transform(question_info['char_id']).todense() )
qChars.columns = map(lambda i: "c" + str(i), range(len(qChars.columns)))

# question_info = pd.concat([question_info, qWords, qChars], axis=1, join_axes=[question_info.index])

question_info = pd.concat([question_info, qChars], axis=1, join_axes=[question_info.index])

# FEATURES = [ "tag", "upvotes", "answers", "top_answers", "answerability" ]
# FEATURES = [ "ease", "popularity", "votability", "answerability" ]
FEATURES = qChars.columns.tolist()
# + qWords.columns.tolist() + qChars.columns.tolist()


km = KMeans(n_clusters=20,
 init='k-means++',
 n_init=10,
 max_iter=100,
 tol=0.0001,
 precompute_distances='auto',
 verbose=0,
 random_state=None,
 copy_x=True,
 n_jobs=1,
 algorithm='auto')


km.fit(question_info[FEATURES])
question_info['clusters'] = km.labels_

res = question_info[['question_id', 'clusters']]
res.to_csv('data/question-clusters.csv', sep=",", index=None)

for c in km.cluster_centers_:
  plt.plot(c)
plt.show()

# for c in range(5):
#   plt.hist(question_info[question_info.clusters == c]['tag'], range(21), stacked=True, normed = True)
# plt.show()
