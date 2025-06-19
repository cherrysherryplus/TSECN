# Put images to be evaluated in the 'results/<method name>/<dataset name>' folder
# Then run this script to evaluate.
# For other datasets, change the dataset name accordingly.
python tools/measure_rf.py TSECN LIME
python tools/measure.py TSECN LOL-v1-real
