# python feature_reduction.py uT 1 0.1
# python feature_reduction.py uT 3 0.1
# python feature_reduction.py uT 5 0.1
# python feature_reduction.py uT 7 0.1
# python feature_reduction.py uT 9 0.1
# python feature_reduction.py uT 11 0.1
# python feature_reduction.py uT 13 0.1
# python feature_reduction.py uT 15 0.1
# python feature_reduction.py uT 17 0.1
# python feature_reduction.py uT 19 0.1
# python feature_reduction.py uT 21 0.1
# python feature_reduction.py uT 23 0.1
# python feature_reduction.py uT 25 0.1
# python feature_reduction.py uT 27 0.1
# python feature_reduction.py uT 29 0.1
# python feature_reduction.py uT 31 0.1
# python feature_reduction.py uT 33 0.1
# python feature_reduction.py uT 35 0.1
# python feature_reduction.py uT 37 0.1
# python feature_reduction.py uT 39 0.1
# python feature_reduction.py uT 41 0.1
# python feature_reduction.py uT 43 0.1
# python feature_reduction.py uT 45 0.1
# python feature_reduction.py uT 47 0.1
# python feature_reduction.py uT 49 0.1
# python feature_reduction.py uT 51 0.1
# python feature_reduction.py uT 53 0.1
# python feature_reduction.py uT 55 0.1
# python feature_reduction.py uT 57 0.1
python feature_reduction.py uT 59 0.1
python feature_reduction.py uT 61 0.1
python feature_reduction.py uT 63 0.1
python feature_reduction.py uT 65 0.1
python feature_reduction.py uT 67 0.1
python feature_reduction.py uT 69 0.1
python feature_reduction.py uT 71 0.1

python run.py data/train.csv data/validate.csv data/validate.csv.results.mf1 MF "[11, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf2 MF "[15, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf3 MF "[17, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf4 MF "[21, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf5 MF "[25, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf6 MF "[29, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf7 MF "[33, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf8 MF  "[37, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf9 MF  "[41, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf10 MF "[45, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf11 MF "[49, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf12 MF "[53, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf13 MF "[57, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf14 MF "[61, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf15 MF "[63, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf16 MF "[67, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf17 MF "[71, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf18 MF "[73, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf19 MF "[77, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf20 MF "[81, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf21 MF "[83, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf22 MF "[87, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf23 MF "[91, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf24 MF "[93, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf25 MF "[97, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf26 MF "[101, 0.1, 0]"
wait

python run.py data/train.csv data/validate.csv data/validate.csv.results.mf1 MF "[41, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf2 MF "[42, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf3 MF "[43, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf4 MF "[44, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf5 MF "[45, 0.1, 0]"


python run.py data/train.csv data/validate.csv data/validate.csv.results.mf1 MF "[45, 0.001, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf2 MF "[45, 0.005, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf3 MF "[45, 0.01, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf4 MF "[45, 0.05, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf5 MF "[45, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf6 MF "[45, 0.5, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf7 MF "[45, 1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf8 MF "[45, 5, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf9 MF "[45, 10, 0]"


python run.py data/train.csv data/validate.csv data/validate.csv.results.mf1 MF "[45, 0.1, 0]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf2 MF "[45, 0.1, -0.0001]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf3 MF "[45, 0.1, -0.0003]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf4 MF "[45, 0.1, -0.0005]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf5 MF "[45, 0.1, -0.0009]"
python run.py data/train.csv data/validate.csv data/validate.csv.results.mf6 MF "[45, 0.1, -0.001]"


python run.py data/train.csv data/validate.csv data/validate.csv.results.itemCF7 ItemCf "[51, 0, 'hamming']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.itemCF6 ItemCf "[37, 0, 'hamming']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.itemCF5 ItemCf "[23, 0, 'hamming']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.itemCF4 ItemCf "[11, 0, 'hamming']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.itemCF3 ItemCf "[9, 0, 'hamming']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.itemCF2 ItemCf "[5, 0, 'hamming']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.itemCF1 ItemCf "[1, 0, 'hamming']"

python run.py data/train.csv data/validate.csv data/validate.csv.results.userCF7 UserCf "[51, -0.0005, 'euclidean']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.userCF6 UserCf "[37, -0.0005, 'euclidean']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.userCF5 UserCf "[23, -0.0005, 'euclidean']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.userCF4 UserCf "[11, -0.0005, 'euclidean']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.userCF3 UserCf "[9, -0.0005, 'euclidean']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.userCF2 UserCf "[5, -0.0005, 'euclidean']"
python run.py data/train.csv data/validate.csv data/validate.csv.results.userCF1 UserCf "[1, -0.0005, 'euclidean']"



python run.py data/train.csv data/train.csv data/1.csv MF "[45, 0.1, -0.0005]"
python run.py data/train.csv data/train.csv data/2.csv ItemCf "[51, 0, 'hamming']"
python run.py data/train.csv data/train.csv data/3.csv ItemCf "[37, 0, 'hamming']"
python run.py data/train.csv data/train.csv data/4.csv ItemCf "[23, 0, 'hamming']"
python run.py data/train.csv data/train.csv data/5.csv ItemCf "[11, 0, 'hamming']"
python run.py data/train.csv data/train.csv data/6.csv ItemCf "[9, 0, 'hamming']"
python run.py data/train.csv data/train.csv data/7.csv ItemCf "[5, 0, 'hamming']"
python run.py data/train.csv data/train.csv data/8.csv ItemCf "[1, 0, 'hamming']"
python run.py data/train.csv data/train.csv data/9.csv UserCf "[51, -0.0005, 'euclidean']"
python run.py data/train.csv data/train.csv data/10.csv UserCf "[37, -0.0005, 'euclidean']"
python run.py data/train.csv data/train.csv data/11.csv UserCf "[23, -0.0005, 'euclidean']"
python run.py data/train.csv data/train.csv data/12.csv UserCf "[11, -0.0005, 'euclidean']"
python run.py data/train.csv data/train.csv data/13.csv UserCf "[9, -0.0005, 'euclidean']"
python run.py data/train.csv data/train.csv data/14.csv UserCf "[5, -0.0005, 'euclidean']"
python run.py data/train.csv data/train.csv data/15.csv UserCf "[1, -0.0005, 'euclidean']"
