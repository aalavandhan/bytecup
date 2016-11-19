import sys
import os

import numpy as np
import pandas as pd
import sys
from util.eval import generate_ndcg_scores

ARGUMENTS = sys.argv[1]
OP        = sys.argv[2]

output =  open(OP, "a")

nScores = [ ]

for i in range(10):
  train = "kfold/train-{0}.csv".format(i)
  validate  = "kfold/validate-{0}.csv".format(i)

  os.system("python run.py {0} {1} results.csv ".format(train, validate) + ARGUMENTS)



  results = generate_ndcg_scores(validate, "results.csv")
  scores  = map(lambda s: s[0], results)

  nScores.append( np.array(scores).mean() * 100 )

output.write("'{0}', {1}".format(ARGUMENTS, np.array(nScores).mean()))
output.close()

