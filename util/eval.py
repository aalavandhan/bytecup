from rank_metrics import ndcg_at_k
from dict_reader import DictReader


def generate_ndcg_scores(TRUTH, EST):
  truth = DictReader(TRUTH).load()
  est   = DictReader(EST).load()

  # ndcg score
  scores = [ ]

  for q in est.keys():
    recommended = sorted(est[q].keys(), key=lambda u: est[q][u], reverse=True)
    recommendation = map(lambda u: int(truth[q][u]),recommended)

    if not any(recommendation):
      pass
    else:
      s = ndcg_at_k(recommendation, 5, method=1) * 0.5 + ndcg_at_k(recommendation, 10, method=1) * 0.5
      scores.append((s, q))

  return scores
