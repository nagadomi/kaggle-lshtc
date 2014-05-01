CXXFLAGS=-std=c++0x -fopenmp -funroll-loops -march=native -Wno-unused-function -D_GLIBCXX_PARALLEL -Ofast -g -Wall -DNDEBUG 
CXX=g++

all: prefetch train predict vt_ncc vt_knn vt_prefetch vt_train vt_classifier validation knn ncc

clean:
	rm -fr prefetch train predict vt_ncc vt_knn vt_prefetch vt_train vt_classifier validation knn ncc

prefetch: prefetch.cpp reader.hpp tick.hpp util.hpp  inverted_index.hpp tfidf_transformer.hpp ncc_cache.hpp  nearest_centroid_classifier.hpp SETTINGS.h
	$(CXX) prefetch.cpp -o prefetch -DVALIDATION_TEST=0 $(CXXFLAGS)

train: train.cpp reader.hpp tick.hpp util.hpp tfidf_transformer.hpp ncc_cache.hpp classifier_storage.hpp SETTINGS.h
	$(CXX) train.cpp -o train -DVALIDATION_TEST=0 $(CXXFLAGS)

predict: predict.cpp  reader.hpp tick.hpp util.hpp inverted_index.hpp tfidf_transformer.hpp nearest_centroid_classifier.hpp classifier_storage.hpp binary_classifier.hpp  SETTINGS.h

vt_train: train.cpp reader.hpp tick.hpp util.hpp tfidf_transformer.hpp ncc_cache.hpp classifier_storage.hpp SETTINGS.h
	$(CXX) train.cpp -o vt_train -DVALIDATION_TEST=1 $(CXXFLAGS)

vt_knn: vt_knn.cpp reader.hpp tick.hpp util.hpp inverted_index.hpp tfidf_transformer.hpp evaluation.hpp SETTINGS.h

vt_ncc: vt_ncc.cpp reader.hpp tick.hpp util.hpp inverted_index.hpp tfidf_transformer.hpp nearest_centroid_classifier.hpp evaluation.hpp SETTINGS.h

vt_prefetch: prefetch.cpp reader.hpp tick.hpp util.hpp  inverted_index.hpp tfidf_transformer.hpp ncc_cache.hpp  nearest_centroid_classifier.hpp SETTINGS.h
	$(CXX) prefetch.cpp -o vt_prefetch -DVALIDATION_TEST=1 $(CXXFLAGS)

vt_classifier: vt_classifier.cpp reader.hpp tick.hpp util.hpp  inverted_index.hpp tfidf_transformer.hpp ncc_cache.hpp binary_classifier.hpp SETTINGS.h
	$(CXX) vt_classifier.cpp -o vt_classifier -DVALIDATION_TEST=1 $(CXXFLAGS)

validation: validation.cpp reader.hpp tick.hpp util.hpp inverted_index.hpp tfidf_transformer.hpp nearest_centroid_classifier.hpp classifier_storage.hpp binary_classifier.hpp SETTINGS.h
	$(CXX) validation.cpp -o validation -DVALIDATION_TEST=1 $(CXXFLAGS)

knn: knn.cpp reader.hpp tick.hpp util.hpp inverted_index.hpp tfidf_transformer.hpp SETTINGS.h

ncc: ncc.cpp reader.hpp tick.hpp util.hpp inverted_index.hpp tfidf_transformer.hpp SETTINGS.h
