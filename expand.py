import numpy as np
import pandas as pd
import scipy

import libpmf.libpmf as libpmf
from scipy import sparse

user_info = pd.read_csv("data/user_info.txt", sep="\t", header=None, names=[
    "user_id", "expert_tags", "word_id", "char_id"
])
question_info = pd.read_csv("data/question_info.txt", sep="\t", header=None, names=[
    "question_id", "tag", "word_id", "char_id", "upvotes", "answers", "top_answers"
])

train_info = pd.read_csv("data/invited_info_train.txt", sep="\t", header=None, names=[
    "question_id", "user_id", "answered"
])

safeInt = lambda v: int(v) if v != '' else None
cleanFeature = lambda r: map(safeInt, r.split('/'))

# single time computation
find_max = lambda v: max(cleanFeature(v))
find_min = lambda v: min(cleanFeature(v))

# Computing the parameter ranges to vectorize user parameters
USER_TAG = [0, 142]
# max_expert_tag = user_info['expert_tags'].apply(find_max).max()
# min_expert_tag = user_info['expert_tags'].apply(find_min).min()

USER_WORD_ID = [0, 37809]
# max_word_id = user_info['word_id'].apply(find_max).max()
# min_word_id = user_info['word_id'].apply(find_max).min()

USER_CHAR_ID = [0, 4021]
# max_char_id = user_info['char_id'].apply(find_max).max()
# min_char_id = user_info['char_id'].apply(find_max).max()

QUESTION_TAG = [0, 19]
# max_question_tag = question_info['tag'].max()
# min_question_tag = question_info['tag'].min()

QUESTION_WORD_ID = [0, 13230]
# max_word_id = question_info['word_id'].apply(find_max).max()
# min_word_id = question_info['word_id'].apply(find_max).max()

QUESTION_CHAR_ID = [0, 2958]
# max_char_id = question_info['char_id'].apply(find_max).max()
# min_char_id = question_info['char_id'].apply(find_max).max()

question_info['ease'] = ( question_info['answers'] - question_info['answers'].min()  ) / question_info['answers'].max()
question_info['popularity'] = ( question_info['top_answers'] - question_info['top_answers'].min() ) / question_info['top_answers'].max()
question_info['votability'] = ( question_info['upvotes'] - question_info['upvotes'].min() ) / question_info['upvotes'].max()
question_info['nTag'] = ( question_info['tag'] - question_info['tag'].min() ) / question_info['tag'].max()
question_info['answerability'] = ( question_info['top_answers'] / question_info['answers'] )
question_info['answerability'] =  question_info['answerability'].fillna(0)

# ASKED vs ANSWERED
answered = train_info[ train_info.answered == 1 ].groupby('user_id').count()['answered']
asked = train_info.groupby('user_id').count()['answered'].rename('asked')
user_info = user_info.join(answered, on="user_id", how="left" ).join(asked, on="user_id", how="left" )
user_info['answered'] =  user_info['answered'].fillna(0)
user_info['asked']    =  user_info['asked'].fillna(0)

answered = train_info[ train_info.answered == 1 ].groupby('question_id').count()['answered']
asked = train_info.groupby('question_id').count()['answered'].rename('asked')
question_info = question_info.join(answered, on="question_id", how="left" ).join(asked, on="question_id", how="left")
question_info['answered'] =  question_info['answered'].fillna(0)
question_info['asked']    =  question_info['asked'].fillna(0)

user_info['idx'] = user_info.index
question_info['idx'] = question_info.index


def expand(df):
    row  = [ ]
    col  = [ ]
    data = [ ]
    for i in df.index:
        for e in [x for x in df.ix[i] if x is not None]:
            row.append(i)
            col.append(e)
            data.append(1)
    return (row, col, data)

def vectorize(ds, column):
    splitFeatures = ds[column].apply(cleanFeature)
    return expand(splitFeatures)

def writeResults(fileName, data):
    f = open("data/" + fileName, "w+")
    for d in data:
        f.write(",".join(map(str, d)))
        f.write("\n")
    f.close()

def MF(d, column, f1, f2, k=5):
    (row, col, data) = vectorize(d, column)
    V = scipy.sparse.csr_matrix((data, (row, col)))
    model = libpmf.train(V, '-k {0} -l 0.1 -t {0}'.format(k))
    writeResults(f1, model['W'])
    writeResults(f2, model['H'])

MF(user_info, "expert_tags", 'user-expert-latent', 'expert-user-latent', 34)
MF(user_info, "word_id", 'user-word-latent', 'word-user-latent', 2858)
MF(user_info, "char_id", 'user-char-latent', 'char-user-latent', 739)
MF(question_info, "word_id", 'question-word-latent', 'word-question-latent', 3265)
MF(question_info, "char_id", 'question-char-latent', 'char-question-latent', 434)


question_word_latent = pd.read_csv("data/question-word-latent", sep=",", header=None)
question_word_latent.columns = map(lambda c: "qw" + c, range(len(question_word_latent.columns)))

question_char_latent = pd.read_csv("data/question-char-latent", sep=",", header=None)
question_char_latent.columns = map(lambda c: "qc" + c, range(len(question_char_latent.columns)))

question_info = pd.concat([question_info, question_word_latent, question_char_latent], axis=1, join_axes=[question_info.index])

user_word_latent = pd.read_csv("data/user-word-latent", sep=",", header=None)
user_word_latent.columns = map(lambda c: "uw" + c, range(len(user_word_latent.columns)))

user_char_latent = pd.read_csv("data/user-char-latent", sep=",", header=None)
user_char_latent.columns = map(lambda c: "uc" + c, range(len(user_char_latent.columns)))

user_expert_latent = pd.read_csv("data/user-expert-latent", sep=",", header=None)
user_expert_latent.columns = map(lambda c: "ut" + c, range(len(user_expert_latent.columns)))

user_info = pd.concat([user_info, user_word_latent, user_char_latent, user_expert_latent], axis=1, join_axes=[user_info.index])


user_info.to_csv("data/user-features", sep=",", columns=["user_id"] + user_word_latent.columns + user_char_latent.columns + user_expert_latent.columns)
question_info.to_csv("data/question-features", sep=",", columns=[ "question_id", "tag", "upvotes", "answers", "top_answers" ] + question_word_latent.columns + question_char_latent.columns)
