# python run.py data/train.csv data/validate.csv data/validate.csv.results.u1 UserCf "[11,0]" &
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


python run.py data/train.csv data/validate.csv data/validate.csv.results.uc1 UserCf "[11,-0.0001, 1.01]" &
python run.py data/train.csv data/validate.csv data/validate.csv.results.uc2 UserCf "[11,-0.0001, 1.1]" &
python run.py data/train.csv data/validate.csv data/validate.csv.results.uc3 UserCf "[11,-0.0001, 1.5]" &
python run.py data/train.csv data/validate.csv data/validate.csv.results.uc4 UserCf "[11,-0.0001, 2]" &
python run.py data/train.csv data/validate.csv data/validate.csv.results.uc5 UserCf "[11,-0.0001, 2.5]" &

python run.py data/train.csv data/validate.csv data/validate.csv.results.ic1 ItemCf "[7,-1, 1.01]" &
python run.py data/train.csv data/validate.csv data/validate.csv.results.ic2 ItemCf "[7,-1, 1.1]" &
python run.py data/train.csv data/validate.csv data/validate.csv.results.ic3 ItemCf "[7,-1, 1.5]" &
python run.py data/train.csv data/validate.csv data/validate.csv.results.ic4 ItemCf "[7,-1, 2]" &
python run.py data/train.csv data/validate.csv data/validate.csv.results.ic5 ItemCf "[7,-1, 2.5]" &
wait
