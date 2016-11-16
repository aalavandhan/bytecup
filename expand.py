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

MF(user_info, "expert_tags", 'user-expert-latent', 'expert-user-latent', 10)
MF(user_info, "word_id", 'user-word-latent', 'word-user-latent', 51)
MF(user_info, "char_id", 'user-char-latent', 'char-user-latent', 51)
MF(question_info, "word_id", 'question-word-latent', 'word-question-latent', 51)
MF(question_info, "char_id", 'question-char-latent', 'char-question-latent', 51)


question_word_latent = pd.read_csv("data/question-word-latent", sep=",", header=None, names=[
    "wa1", "wa2", "wa3", "wa4", "wa5", "wa6", "wa7", "wa8", "wa9", "wa10",
    "wb1", "wb2", "wb3", "wb4", "wb5", "wb6", "wb7", "wb8", "wb9", "wb10",
    "wc1", "wc2", "wc3", "wc4", "wc5", "wc6", "wc7", "wc8", "wc9", "wc10",
    "wd1", "wd2", "wd3", "wd4", "wd5", "wd6", "wd7", "wd8", "wd9", "wd10",
    "we1", "we2", "we3", "we4", "we5", "we6", "we7", "we8", "we9", "we10",
    "wf1",
])
question_char_latent = pd.read_csv("data/question-char-latent", sep=",", header=None, names=[
    "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10",
    "cb1", "cb2", "cb3", "cb4", "cb5", "cb6", "cb7", "cb8", "cb9", "cb10",
    "cc1", "cc2", "cc3", "cc4", "cc5", "cc6", "cc7", "cc8", "cc9", "cc10",
    "cd1", "cd2", "cd3", "cd4", "cd5", "cd6", "cd7", "cd8", "cd9", "cd10",
    "ce1", "ce2", "ce3", "ce4", "ce5", "ce6", "ce7", "ce8", "ce9", "ce10",
    "cf1",
])
question_info = pd.concat([question_info, question_word_latent, question_char_latent], axis=1, join_axes=[question_info.index])

user_word_latent = pd.read_csv("data/user-word-latent", sep=",", header=None, names=[
    "wua1", "wua2", "wua3", "wua4", "wua5", "wua6", "wua7", "wua8", "wua9", "wua10",
    "wub1", "wub2", "wub3", "wub4", "wub5", "wub6", "wub7", "wub8", "wub9", "wub10",
    "wuc1", "wuc2", "wuc3", "wuc4", "wuc5", "wuc6", "wuc7", "wuc8", "wuc9", "wuc10",
    "wud1", "wud2", "wud3", "wud4", "wud5", "wud6", "wud7", "wud8", "wud9", "wud10",
    "wue1", "wue2", "wue3", "wue4", "wue5", "wue6", "wue7", "wue8", "wue9", "wue10",
    "wuf1",
])
user_char_latent = pd.read_csv("data/user-char-latent", sep=",", header=None, names=[
    "cua1", "cua2", "cua3", "cua4", "cua5", "cua6", "cua7", "cua8", "cua9", "cua10",
    "cub1", "cub2", "cub3", "cub4", "cub5", "cub6", "cub7", "cub8", "cub9", "cub10",
    "cuc1", "cuc2", "cuc3", "cuc4", "cuc5", "cuc6", "cuc7", "cuc8", "cuc9", "cuc10",
    "cud1", "cud2", "cud3", "cud4", "cud5", "cud6", "cud7", "cud8", "cud9", "cud10",
    "cue1", "cue2", "cue3", "cue4", "cue5", "cue6", "cue7", "cue8", "cue9", "cue10",
    "cuf1",
])
user_expert_latent = pd.read_csv("data/user-expert-latent", sep=",", header=None, names=[
    "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10",
    "t11", "t12", "t13", "t14", "t15",
])
user_info = pd.concat([user_info, user_word_latent, user_char_latent, user_expert_latent], axis=1, join_axes=[user_info.index])


user_info.to_csv("data/user-features", sep=",", columns=[
    "user_id",
    "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10",
    "t11", "t12", "t13", "t14", "t15",
    "wua1", "wua2", "wua3", "wua4", "wua5", "wua6", "wua7", "wua8", "wua9", "wua10",
    "wub1", "wub2", "wub3", "wub4", "wub5", "wub6", "wub7", "wub8", "wub9", "wub10",
    "wuc1", "wuc2", "wuc3", "wuc4", "wuc5", "wuc6", "wuc7", "wuc8", "wuc9", "wuc10",
    "wud1", "wud2", "wud3", "wud4", "wud5", "wud6", "wud7", "wud8", "wud9", "wud10",
    "wue1", "wue2", "wue3", "wue4", "wue5", "wue6", "wue7", "wue8", "wue9", "wue10",
    "wuf1",
    "cua1", "cua2", "cua3", "cua4", "cua5", "cua6", "cua7", "cua8", "cua9", "cua10",
    "cub1", "cub2", "cub3", "cub4", "cub5", "cub6", "cub7", "cub8", "cub9", "cub10",
    "cuc1", "cuc2", "cuc3", "cuc4", "cuc5", "cuc6", "cuc7", "cuc8", "cuc9", "cuc10",
    "cud1", "cud2", "cud3", "cud4", "cud5", "cud6", "cud7", "cud8", "cud9", "cud10",
    "cue1", "cue2", "cue3", "cue4", "cue5", "cue6", "cue7", "cue8", "cue9", "cue10",
    "cuf1",
])
question_info.to_csv("data/question-features", sep=",", columns=[
    "question_id",
    "tag", "upvotes", "answers", "top_answers",
    "nTag", "votability", "ease", "popularity", "answerability",
    "wa1", "wa2", "wa3", "wa4", "wa5", "wa6", "wa7", "wa8", "wa9", "wa10",
    "wb1", "wb2", "wb3", "wb4", "wb5", "wb6", "wb7", "wb8", "wb9", "wb10",
    "wc1", "wc2", "wc3", "wc4", "wc5", "wc6", "wc7", "wc8", "wc9", "wc10",
    "wd1", "wd2", "wd3", "wd4", "wd5", "wd6", "wd7", "wd8", "wd9", "wd10",
    "we1", "we2", "we3", "we4", "we5", "we6", "we7", "we8", "we9", "we10",
    "wf1",
    "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10",
    "cb1", "cb2", "cb3", "cb4", "cb5", "cb6", "cb7", "cb8", "cb9", "cb10",
    "cc1", "cc2", "cc3", "cc4", "cc5", "cc6", "cc7", "cc8", "cc9", "cc10",
    "cd1", "cd2", "cd3", "cd4", "cd5", "cd6", "cd7", "cd8", "cd9", "cd10",
    "ce1", "ce2", "ce3", "ce4", "ce5", "ce6", "ce7", "ce8", "ce9", "ce10",
    "cf1",
])
