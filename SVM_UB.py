import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import preprocessing as p
from sklearn.svm import SVC

def tokenize(text):
	return str(text).strip().split("/")

if __name__ == "__main__":

	countVector = CountVectorizer(tokenizer = tokenize, lowercase=False)
	##################### QUES_FEATURES #####################
	quesFile = "bytecup2016data/question_info.txt"
	colName = ["Q_ID", "Q_TAG", "Q_WORD_ID", "Q_CHAR_ID", "NO_OF_UPVOTES", "NO_OF_ANS", "NO_OF_TOP_QUALITY_ANS"]
	quesDF = pd.read_csv(quesFile, sep = "\t", header=None, names=colName)
	
	tagQMatrix = countVector.fit_transform(quesDF["Q_TAG"]).toarray()
	tagDF = pd.DataFrame(tagQMatrix, columns = countVector.get_feature_names())
	charQMatrix = countVector.fit_transform(quesDF["Q_CHAR_ID"]).toarray()
	charDF = pd.DataFrame(charQMatrix, columns = countVector.get_feature_names())
	wordQMatrix = countVector.fit_transform(quesDF["Q_WORD_ID"]).toarray()
	wordDF = pd.DataFrame(wordQMatrix, columns = countVector.get_feature_names())

	quesDF = pd.concat([quesDF, tagDF, charDF], axis=1, join='inner')
	quesDF.drop("Q_WORD_ID",axis=1, inplace=True)
	quesDF.drop("Q_CHAR_ID",axis=1, inplace=True)
	quesDF.drop("Q_TAG",axis=1, inplace=True)
	print quesDF

	###################### EXPERT_FEATURES #####################
	# userFile = "bytecup2016data/user_info.txt"
	# colName = ["E_ID", "E_TAG", "E_WORD_ID", "E_CHAR_ID"]
	# userDF = pd.read_csv(userFile, sep = "\t", header=None, names=colName)

	# tagEMatrix = countVector.fit_transform(userDF["E_TAG"]).toarray()
	# tagDF = pd.DataFrame(tagEMatrix, columns = countVector.get_feature_names())
	# charEMatrix = countVector.fit_transform(userDF["E_CHAR_ID"]).toarray()
	# charDF = pd.DataFrame(charEMatrix, columns = countVector.get_feature_names())
	# wordEMatrix = countVector.fit_transform(userDF["E_WORD_ID"]).toarray()
	# wordDF = pd.DataFrame(wordEMatrix, columns = countVector.get_feature_names())

	# userDF = pd.concat([userDF, tagDF], axis=1, join='inner')
	# userDF.drop("E_WORD_ID",axis=1, inplace=True)
	# userDF.drop("E_CHAR_ID",axis=1, inplace=True)
	# userDF.drop("E_TAG",axis=1, inplace=True)
	# print userDF

	######################## TRAIN MODEL ########################
	trainFile = "bytecup2016data/invited_info_train.txt"
	colName = ["Q_ID", "E_ID", "LABEL"]
	trainData = pd.read_csv(trainFile, sep = "\t", header=None, names=colName)

	trainData = trainData.reset_index().merge(quesDF, on="Q_ID", how="inner", sort=False).sort_values("index")
	trainData.drop("index", axis =1, inplace=True)
	trainData = trainData.reset_index()
	# trainData = trainData.merge(userDF, on="E_ID", how="inner", sort=False).sort_values("index")
	# trainData.drop("index", axis =1, inplace=True)
	# trainData = trainData.reset_index()
	# print trainData
	# print trainData["E_ID"][1]

	###################### TEST READING ######################
	testFile = "bytecup2016data/validate_nolabel.txt"
	colName = ["Q_ID", "E_ID", "LABEL"]
	testData = pd.read_csv(testFile, header=0, names=colName)
	testData.drop("LABEL",axis=1, inplace=True)
	testData = testData.reset_index().merge(quesDF, on="Q_ID", how="inner", sort=False).sort_values("index")
	testData.drop("index", axis=1, inplace=True)
	testData = testData.reset_index()
	# testData = testData.reset_index().merge(userDF, on="E_ID", how="inner", sort=False).sort_values("index")
	# testData.drop("index", axis=1, inplace=True)
	# testData = testData.reset_index()
	
	################### USER BASED ######################################

	probList = np.zeros(testData.index.size)
	for i in range(testData.index.size):
		if probList[i] == 0:
			expId = testData["E_ID"][i]
			df = (trainData.loc[trainData["E_ID"] == expId]).copy()
			#print df
			if df.empty:
				probList[i] = float(0.0)
				continue
			model = SVC(probability=True)
			Y = df["LABEL"].tolist()
			if not(0 in Y and 1 in Y):
				if 0 in Y:
					probList[i] = 0.0
				elif 1 in Y:
					probList[i] = 1.0
				continue
			df.drop("Q_ID", axis = 1,inplace = True)
			df.drop("E_ID", axis = 1,inplace = True)
			df.drop("LABEL", axis = 1,inplace = True)
			X = df.as_matrix()
			# X = p.MinMaxScaler().fit_transform(np.float32(X))
			model = model.fit(X, Y)
			testDF = (testData.loc[testData["E_ID"] == expId]).copy()
			testDF.drop("Q_ID", axis = 1,inplace = True)
			testDF.drop("E_ID", axis = 1,inplace = True)
			testX = testDF.as_matrix()
			#print testX
			# testX = p.MinMaxScaler().fit_transform(np.float32(testX))
			testY = model. predict_proba(testX)
			k = 0
			for j in testDF.index:
				probList[j] = testY[k][1]
				k = k+1
	

	print probList
	labelFile = open("train_validate.txt","w")
	labelFile.write("qid,uid,label\n")
	for index,i in enumerate(probList):
		labelFile.write(str(testData["Q_ID"][index]) +"," + str(testData["E_ID"][index]) + "," +str(i)+"\n")
	labelFile.close()

	
