import numpy as np
import sys

import matplotlib.pyplot as plt

from util.eval import generate_ndcg_scores

TRUTH   = sys.argv[1]
MODEL1  = sys.argv[2]
MODEL2  = sys.argv[3]

r1 = generate_ndcg_scores(TRUTH, MODEL1)
r2 = generate_ndcg_scores(TRUTH, MODEL2)

# GOOD
s1  = filter(lambda s: s[0] >= 0.5, r1)
s2  = filter(lambda s: s[0] >= 0.5, r2)
q1 = map(lambda s: s[1], s1)
q2 = map(lambda s: s[1], s2)
it = set(q1) - set(q2)
un = set(q1).union( set(q2) )
print "GOOD : {0}%".format(float(len(it)) / float(len(r1)) * 100)

# BAD
s1  = filter(lambda s: s[0] == 0, r1)
s2  = filter(lambda s: s[0] == 0, r2)
q1 = map(lambda s: s[1], s1)
q2 = map(lambda s: s[1], s2)
it = set(q1) - set(q2)
un = set(q1).union( set(q2) )
print "BAD : {0}%".format(float(len(it)) / float(len(r1)) * 100)
