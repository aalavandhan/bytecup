# Running models

```
python run.py [TRAIN_PATH] [TEST_PATH] [RESULT_PATH] [MODEL] [HYPER PARAMETERS]
```

MODELS = [
  UserCf,
  ItemCf,
  MF,
  MFImp,
  MFGLab,
  MFContentGLab,
  UserCfGLab,
  ItemCfGLab,
  UserCfContentGLab,
  ItemCfContentGLab,
  TextSimilarity,
  TagRule,
]

HYPER_PARAMTERS are specific to a model lambda, distance, UB (etc)

```
example: python run.py data/train.csv data/validate.csv data/validate.csv.results.mf1 MF "[11, 0.1, 0]"
```

# Ensemble
```
python dtree.py [TRAIN_SET] [TEST_SET] [TRAIN] [TEST] [RESULT_PATH]
```
