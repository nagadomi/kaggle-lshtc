#ifndef SETTINGS_H
#define SETTINGS_H

#define K_TRAIN   10
#define K_PREDICT 8
#define TRAIN_DATA "../data/train.csv"
#define TEST_DATA "../data/test.csv"

#if VALIDATION_TEST
#  define CACHE       "./vt_ncc_cache.bin"
#  define CACHE_TEST  "./vt_ncc_cache_test.bin"
#  define CENTROID    "./vt_centroid.bin"
#  define WEIGHT      "./vt_weight.bin"
#  define MODEL       "./vt_model.bin"
#else
#  define CACHE       "./ncc_cache.bin"
#  define CENTROID    "./centroid.bin"
#  define WEIGHT      "./weight.bin"
#  define MODEL       "./model.bin"
#endif
#define SUBMISSION    "./submission.txt"

#define VT_SEED     13

/* parameter for binary classifier */
#define LR_ETA         0.2f
#define LR_P           0.76f
#define LR_ITERATION   40

#endif
