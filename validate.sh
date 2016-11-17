# python run.py data/train.csv data/validate.csv data/validate  .csv.results.u1 UserCf "[11,0]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.u2 UserCf "[11,-1]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.u3 UserCf "[11,-0.5]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.u4 UserCf "[11,-0.1]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.u5 UserCf "[11,-0.01]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.u6 UserCf "[11,-0.001]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.u7 UserCf "[11,-0.0001]" &

# python run.py data/train.csv data/validate.csv data/validate.csv.results.i1 ItemCf "[7,0]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.i2 ItemCf "[7,-1]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.i3 ItemCf "[7,-0.5]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.i4 ItemCf "[7,-0.1]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.i5 ItemCf "[7,-0.01]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.i6 ItemCf "[7,-0.001]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.i7 ItemCf "[7,-0.0001]" &
# wait


# python run.py data/train.csv data/validate.csv data/validate.csv.results.uinf1 UserCfInf "[11,-0.0001]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.iinf1 ItemCfInf "[7,-1]" &
# wait

# python run.py data/train.csv data/validate.csv data/validate.csv.results.mf1 MF "[27, 0.1, 0]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.mf2 MF "[27, 0.1, -1]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.mf3 MF "[27, 0.1, -0.5]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.mf4 MF "[27, 0.1, -0.1]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.mf5 MF "[27, 0.1, -0.01]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.mf6 MF "[27, 0.1, -0.001]" &
# python run.py data/train.csv data/validate.csv data/validate.csv.results.mf7 MF "[27, 0.1, -0.0001]" &
# wait

python run.py data/train.csv data/validate.csv data/validate.csv.results.mf1 MFImp "[27, 0.1, 0, 0.0001]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf2 MFImp "[27, 0.1, 0, 0.0005]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf3 MFImp "[27, 0.1, 0, 0.001]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf5 MFImp "[27, 0.1, 0, 0.01]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf6 MFImp "[27, 0.1, 0, 0.05]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf7 MFImp "[27, 0.1, 0, 0.1]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf8 MFImp "[27, 0.1, 0, 0.25]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf9 MFImp "[27, 0.1, 0, 0.5]"


python run.py data/train-full.csv data/test.csv test/r1.csv UserCf  "[11, -0.001, 'euclidean']"
python run.py data/train-full.csv data/test.csv test/r2.csv ItemCf "[7, 0, 'hamming']"
python run.py data/train-full.csv data/test.csv test/r3.csv MF  "[27, 0.1, 0]"
python run.py data/train-full.csv data/test.csv test/r4.csv TagRule
python run.py data/train-full.csv data/test.csv test/r5.csv TextSimilarity "['word']"
python run.py data/train-full.csv data/test.csv test/r6.csv TextSimilarity "['char']"


python run.py data/train-full.csv data/train-full.csv combine/r1.csv UserCf  "[11, -0.001, 'euclidean']"
python run.py data/train-full.csv data/train-full.csv combine/r2.csv ItemCf "[7, 0, 'hamming']"
python run.py data/train-full.csv data/train-full.csv combine/r3.csv MF  "[27, 0.1, 0]"
python run.py data/train-full.csv data/train-full.csv combine/r4.csv TagRule
python run.py data/train-full.csv data/train-full.csv combine/r5.csv TextSimilarity "['word']"
python run.py data/train-full.csv data/train-full.csv combine/r6.csv TextSimilarity "['char']"

