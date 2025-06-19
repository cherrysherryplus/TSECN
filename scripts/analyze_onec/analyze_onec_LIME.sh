# The previous weights are lossed accidentally, so we find a new set of weights
# This new configuration achieves better performance
# The original one is 3.548/10.400/24.575.
# The new one is 3.527/10.371/24.478.
python tools/analyze_onec.py TSECN \
    --which_dataset LIME \
    --save_dir_name TSECN  \
    --wNIQE 15.0 \
    --wBRISQUE 1.0 \
    --wILNIQE 1.5 \
    --wExposure 1.0
    