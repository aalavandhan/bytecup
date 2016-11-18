def writeRecommendationsToFile(recommendations, test_info, TEST_PATH, OUTPUT_PATH):
  res = test_info[['question_id', 'user_id']].copy()
  res['answered'] = recommendations
  res.to_csv(OUTPUT_PATH, sep=",", index=None, header=True)
