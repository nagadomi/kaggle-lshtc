#include "util.hpp"
#include "reader.hpp"
#include "tick.hpp"
#include "tfidf_transformer.hpp"
#include "evaluation.hpp"
#include "ncc_cache.hpp"
#include "binary_classifier.hpp"
#include "classifier_storage.hpp"
#include <cstdio>
#include <map>
#include "SETTINGS.h"

void
build_train_data(category_index_t &dataset,
				 std::vector<fv_t> &data,
				 std::vector<label_t> &labels,
				 NCCCache &cache)
{
	dataset.clear();
	for (int i = 0; i < (int)data.size(); ++i) {
		std::vector<int> results;
		std::set<int> hit_labels;
		if (cache.get(i, results)) {
			if (K_TRAIN < results.size()) {
				results.erase(results.begin() + K_TRAIN, results.end());
			}
			for (auto res = results.begin(); res != results.end(); ++res) {
				hit_labels.insert(*res);
			}
			for (auto l = hit_labels.begin(); l != hit_labels.end(); ++l) {
				auto d = dataset.find(*l);
				if (d != dataset.end()) {
					d->second.push_back(i);
				} else {
					std::vector<int> vec;
					vec.push_back(i);
					dataset.insert(std::make_pair(*l, vec));
				}
			}
		}
	}
}

void
get_train_data(
	int target,
	std::vector<fv_t> &posi,
	std::vector<fv_t> &nega,
	const std::vector<fv_t> &test_data,
	const std::vector<label_t> &test_labels,
	const category_index_t &dataset)
{
	posi.clear();
	nega.clear();
	
	auto target_dataset = dataset.find(target);
	if (target_dataset == dataset.end()) {
		return;
	}
	for (auto i = target_dataset->second.begin(); i != target_dataset->second.end(); ++i) {
		if (test_labels[*i].find(target) != test_labels[*i].end()) {
			posi.push_back(test_data[*i]);
		} else {
			nega.push_back(test_data[*i]);
		}
	}
}

int main(void)
{
	DataReader reader;	
	std::vector<fv_t> data;
	std::vector<label_t> labels;
#if VALIDATION_TEST	
	std::vector<fv_t> test_data;
	std::vector<label_t> test_labels;
#endif
	TFIDFTransformer transformer;
	category_index_t category_index;
	category_index_t dataset;
	long t = tick();
	NCCCache cache;
	ClassifierStorage classifiers;
	
	if (!reader.open(TRAIN_DATA)) {
		fprintf(stderr, "cant read file\n");
		return -1;
	}
	reader.read(data, labels);
	printf("read %ld, %ld, %ldms\n", data.size(), labels.size(), tick() - t);
	reader.close();
	
	if (!cache.load(CACHE)) {
		std::fprintf(stderr, "load failed: %s: please either run ./vt_prefetch\n", CACHE);
		return -1;
	}
	printf("read %ld, %ld, %ldms\n", data.size(), labels.size(), tick() - t);
	
	t = tick();
	build_category_index(category_index, data, labels);
#if VALIDATION_TEST
	srand(VT_SEED);
	split_data(test_data, test_labels, data, labels, category_index, 0.05f);
	build_category_index(category_index, data, labels);
#endif
	transformer.load(WEIGHT);
	transformer.transform(data);
	
	printf("build index %ldms\n", tick() -t );
	t = tick();
	
	build_train_data(dataset, data, labels, cache);
	printf("build dataset %ld %ldms\n", dataset.size(), tick() -t );

	std::vector<std::pair<int, const std::vector<int> *> > category_data;
	for (auto docs = category_index.begin(); docs != category_index.end(); ++docs) {
		category_data.push_back(std::make_pair(docs->first, &docs->second));
	}
	std::random_shuffle(category_data.begin(), category_data.end());
	
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
	for (int i = 0; i < (int)category_data.size(); ++i) {
		if (i % 10000 == 0) {
#ifdef _OPENMP
#pragma omp critical
#endif
			{
				printf("- train %d/%ld %ldms\n",
					   i, category_data.size(), tick() - t);
				t = tick();
			}
		}
		std::vector<fv_t> posi;
		std::vector<fv_t> nega;
		BinaryClassifier model;
		
		get_train_data(category_data[i].first, posi, nega, data, labels, dataset);
		model.train(posi, nega, LR_ETA, LR_P, LR_ITERATION);
		classifiers.set(category_data[i].first, model);
	}
	classifiers.save(MODEL);
	
	return 0;
}
