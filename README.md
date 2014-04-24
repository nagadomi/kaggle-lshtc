code for Large Scale Hierarchical Text Classification competition.

http://www.kaggle.com/c/lshtc

# Summary

a centroid-based flat classifier.

1. Extract K-candidate categories by Nearest Centroid Classifier.
2. Remove some missclassified categories by Binary Classifier from candidate categories.

# Requirements

- Ubuntu 13.10
- g++ 4.8.1
- make
- 32GB RAM

# How to Generate the Solution

please edit SETTINGS.h first.

    make
    ./prefetch
    ./train
    ./predict

NOTE: ./prefetch is very slow. probably processing time exceeds 15 hours.

# MISC programs

## Running the Validation Test

    ./vt_prefech
    ./vt_train
    ./validation

## Running the Simple k-NN baseline

    ./vt_knn

## Running the Simple Centroid Classifier baseline

    ./vt_ncc
