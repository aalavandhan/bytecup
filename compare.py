import numpy as np
import pandas as pd
import sys

# import matplotlib.pyplot as plt
# plt.style.use('ggplot')

from util.eval import generate_ndcg_scores


TRUTH   = sys.argv[1]
EST     = sys.argv[2]

results = generate_ndcg_scores(TRUTH, EST)
scores  = map(lambda s: s[0], results)

print np.array(scores).mean() * 100
# plt.hist(scores, bins=10)
# plt.show()
