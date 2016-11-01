def writeRecommendationsToFile(recommendations, test_info, TEST_PATH):
  res = test_info[['question_id', 'user_id']].copy()
  res['prediction'] = recommendations
  res.columns = ['qid','uid','label']
  res.to_csv(TEST_PATH + ".results", sep=",", index=None, header=True)
