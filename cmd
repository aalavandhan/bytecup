function validate-model(){
  rm data/validate.csv.results

  echo "Recommending with given model"

  python run.py data/train.csv data/validate.csv data/validate.csv.results $1 $2

  echo "Evaluating recommendation"

  python compare.py data/validate.csv data/validate.csv.results
}
