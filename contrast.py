import numpy as np
import sys

import matplotlib.pyplot as plt

from util.eval import generate_ndcg_scores

TRUTH   = sys.argv[1]
MODELS  = sys.argv[2].split(",")


compare = map(lambda m: generate_ndcg_scores(TRUTH, m), MODELS)

result = [ ]

# GOOD
# correct = lambda s: s[0] > 0.75
# base_correct  = filter(correct, r)

# valid = set(base_correct)

for i in range(len(compare[0])):
  optimal = np.array( map(lambda j: compare[j][i][ 0 ], range(len(compare))) ).max()
  result.append(optimal)

  # c_correct = filter(correct, c)
  # valid = set(c_correct).union(valid)

# print float(len(valid)) / float(len(r)) * 100

# BAD
# s1  = filter(lambda s: s[0] == 0, r1)
# s2  = filter(lambda s: s[0] == 0, r2)
# q1 = map(lambda s: s[1], s1)
# q2 = map(lambda s: s[1], s2)
# it = set(q1).union( set(q2) ) - set(q1).intersection( set(q2) )
# un = set(q1).union( set(q2) )
# print "BAD : {0}%".format(float(len(it)) / float(len(r1)) * 100)


print np.array(result).mean()
