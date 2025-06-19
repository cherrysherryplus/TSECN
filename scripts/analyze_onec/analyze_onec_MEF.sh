# The previous weights are lossed accidentally, so we find a new set of weights
# The original one is 3.55/12.35/24.16.
# The new one is 3.57/12.47/24.16.
python tools/analyze_onec.py TSECN \
    --which_dataset MEF \
    --save_dir_name TSECN  \
    --wNIQE 5.0 \
    --wBRISQUE 1.5 \
    --wILNIQE 0.8 \
    --wExposure 1.2
    