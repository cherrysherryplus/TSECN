# The previous weights are lossed accidentally, so we find a new set of weights
# This new configuration achieves better performance, but drops slightly in NIQE
# Lower values are better.
# The original one is 2.826/8.455/22.856.
# The new one is 2.837/8.434/22.855.
python tools/analyze_onec.py TSECN \
    --which_dataset DICM \
    --save_dir_name TSECN  \
    --wNIQE 35.0 \
    --wBRISQUE 0.1 \
    --wILNIQE 1.0 \
    --wExposure 0.5
    