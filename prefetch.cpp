#include "util.hpp"
#include "reader.hpp"
#include "tick.hpp"
#include "tfidf_transformer.hpp"
#include "evaluation.hpp"
#include "nearest_centroid_classifier.hpp"
#include "ncc_cache.hpp"
#include <cstdio>
#include "SETTINGS.h"

int main(int argc, char **argv)
{
	DataReader reader;
	std::vector<fv_t> data;
	std::vector<label_t> labels;
	NearestCentroidClassifier centroid;
	TFIDFTransformer transformer;
	category_index_t category_index;
	long t = tick();
	NCCCache cache;
#if VALIDATION_TEST	
	NCCCache cache_test;
	std::vector<fv_t> test_data;
	std::vector<label_t> test_labels;
#endif
	
	if (!reader.open(TRAIN_DATA)) {
		fprintf(stderr, "cant read file\n");
		return -1;
	}
	reader.read(data, labels);
	printf("read %ld, %ld, %ldms\n", data.size(), labels.size(), tick() - t);
	
	reader.close();
	
	t = tick();
	build_category_index(category_index, data, labels);
#if VALIDATION_TEST
	srand(VT_SEED);
	split_data(test_data, test_labels, data, labels, category_index, 0.05);
	build_category_index(category_index, data, labels);
#endif
	t = tick();
	transformer.train(data);
	transformer.transform(data);
#if VALIDATION_TEST
	transformer.transform(test_data);
#endif
	centroid.train(category_index, data);
	printf("build index %ldms\n", tick() -t );
	t = tick();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
	for (int i = 0; i < (int)data.size(); ++i) {
		std::vector<int> results;
		
		centroid.predict(results, K_TRAIN, data[i]);
		cache.set(i, results);
		if (i % 10000 == 0) {
#ifdef _OPENMP
#pragma omp critical
#endif
			{
				printf("%s: %d/%ld %ldms\n", argv[0], i, data.size(), tick() - t);
				t = tick();
			}
		}
	}
	cache.save(CACHE);

#if VALIDATION_TEST
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
	for (int i = 0; i < (int)test_data.size(); ++i) {
		std::vector<int> results;
		centroid.predict(results, K_TRAIN, test_data[i]);
		cache_test.set(i, results);
		if (i % 10000 == 0) {
#ifdef _OPENMP
#pragma omp critical
#endif
			{
				printf("%s: %d/%ld %ldms\n", argv[0], i, test_data.size(), tick() - t);
				t = tick();
			}
		}
	}
	cache_test.save(CACHE_TEST);
#endif
	centroid.save(CENTROID);
	transformer.save(WEIGHT);
	
	return 0;
}
