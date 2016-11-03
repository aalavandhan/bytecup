import numpy as np
import pandas as pd
import sys

from util.rank_metrics import ndcg_at_k
from util.dict_reader import DictReader

TRUTH   = sys.argv[1]
EST     = sys.argv[2]

truth = DictReader(TRUTH).load()
est   = DictReader(EST).load()

# ndcg score
scores = [ ]

for q in est.keys():
  recommended = sorted(est[q].keys(), key=lambda u: est[q][u], reverse=True)
  recommendation = map(lambda u: truth[q][u],recommended)
  s = ndcg_at_k(recommendation, 5, method=1) * 0.5 + ndcg_at_k(recommendation, 10, method=1) * 0.5
  scores.append(s)

print "NDCG SCORE : {0}%".format( np.array(scores).mean() * 100 )
# MSE
# print "MSE : {0}" .format( ((ip['answered'] - op['answered']) ** 2).mean() * 100 )
