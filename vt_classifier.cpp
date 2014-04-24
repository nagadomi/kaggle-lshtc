#include "util.hpp"
#include "reader.hpp"
#include "tick.hpp"
#include "tfidf_transformer.hpp"
#include "evaluation.hpp"
#include "ncc_cache.hpp"
#include "binary_classifier.hpp"
#include <cstdio>
#include <map>
#include "SETTINGS.h"

// validation program for binary classifier 

void
build_train_data(unsigned int k,
				 category_index_t &dataset,
				 std::vector<fv_t> &data,
				 std::vector<label_t> &labels,
				 NCCCache &cache)
{
	dataset.clear();
	for (int i = 0; i < (int)data.size(); ++i) {
		std::vector<int> results;
		std::set<int> hit_labels;
		if (cache.get(i, results)) {
			if (k < results.size()) {
				results.erase(results.begin() + k, results.end());
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

int
main(void)
{
	DataReader reader;
	std::vector<fv_t> data;
	std::vector<label_t> labels;
	std::vector<fv_t> test_data;
	std::vector<label_t> test_labels;
	NCCCache cache;
	NCCCache cache_test;
	TFIDFTransformer transformer;
	category_index_t category_index;
	category_index_t dataset;	
	category_index_t test_category_index;
	category_index_t test_dataset;	
	long t = tick();
	size_t posi_tp = 0;
	size_t nega_tp = 0;
	size_t nega_count = 0;
	size_t posi_count = 0;
	float posi_acc = 0.0f;
	float nega_acc = 0.0f;
	float mar = 0.0f;
	float map = 0.0f;	
	size_t test_count = 0;
	size_t zero_count = 0;
	
	if (!reader.open(TRAIN_DATA)) {
		fprintf(stderr, "cant read file\n");
		return -1;
	}
	reader.read(data, labels);
	printf("read %ld, %ld, %ldms\n", data.size(), labels.size(), tick() - t);
	reader.close();

	if (!cache.load("vt_ncc_cache.bin")) {
		std::fprintf(stderr, "failed: vt_ncc_cache.bin: please either run ./vt_prefetch\n");
		return -1;
	}
	if (!cache_test.load("vt_ncc_cache_test.bin")) {
		std::fprintf(stderr, "failed: vt_ncc_cache.bin: please either run ./vt_prefetch\n");
		return -1;
	}
	
	t = tick();
	build_category_index(category_index, data, labels);
	srand(13);
	split_data(test_data, test_labels, data, labels, category_index, 0.05f);
	build_category_index(category_index, data, labels);
	build_category_index(test_category_index, test_data, test_labels);
	
	t = tick();
	transformer.train(data);
	transformer.transform(data);
	transformer.transform(test_data);

	printf("build index %ldms\n", tick() -t );
	t = tick();

	build_train_data(K_TRAIN, dataset, data, labels, cache);
	build_train_data(K_PREDICT, test_dataset, test_data, labels, cache_test);
	printf("build dataset %ld %ldms\n", dataset.size(), tick() -t );
	
	for (auto i = category_index.begin(); i != category_index.end(); ++i) {
		long tt = tick();
		// learning classifier each labels
		std::vector<fv_t> posi;
		std::vector<fv_t> nega;
		std::vector<fv_t> test_posi;
		std::vector<fv_t> test_nega;

		if (i->second.size() < 2) {
			continue;
		}
		get_train_data(i->first, posi, nega, data, labels, dataset);
		get_train_data(i->first, test_posi, test_nega, test_data, test_labels, test_dataset);
		BinaryClassifier model;
		model.train(posi, nega, LR_ETA, LR_P, LR_ITERATION);
		{
			int correct_posi = 0;
			int correct_nega = 0;
			std::vector<fv_t> *instance[2] = {&test_nega, &test_posi};
			for (int k = 0; k < 2; ++k) {
				for (auto j = instance[k]->begin();
					 j != instance[k]->end();
					 ++j)
				{
					float p = model.predict(*j);
					if (p > 0.0f) {
						if (k == 1) {
							correct_posi += 1;
						}
					} else {
						if (k == 0) {
							correct_nega += 1;
						}
					}
				}
			}
			posi_count += test_posi.size();
			nega_count += test_nega.size();
			posi_tp += correct_posi;
			nega_tp += correct_nega;
			if (test_posi.size() > 0) {
				posi_acc += (float)correct_posi/test_posi.size();
			} else {
				zero_count += 1;
			}
			if ((correct_posi + (test_nega.size() - correct_nega)) > 0) {
				map += (float)correct_posi / (correct_posi + (test_nega.size() - correct_nega));
			}
			auto c = test_category_index.find(i->first);
			if (c != test_category_index.end() && c->second.size() > 0) {
				mar += (float)correct_posi / test_category_index.find(i->first)->second.size();
			}
			if (test_nega.size() > 0) {
				nega_acc += (float)correct_nega/test_nega.size();
			}
			test_count += 1;
			printf("label %08d: Non zero feature: %ld, accuracy nega:%f%% (%ld), posi:%f%% (%ld) %ldms\n",
				   i->first,
				   model.size(),
				   (float)correct_nega / test_nega.size(),
				   test_nega.size(),
				   (float)correct_posi / test_posi.size(),
				   test_posi.size(),
				   tick() - tt
				);
			printf("posi: %f(%f), nega: %f(%f), P/N: %f, MaF: %f, MaP: %f, MaR: %f zero: %f\n",
				   (float)posi_tp/posi_count,
				   posi_acc/test_count,
				   (float)nega_tp/nega_count,
				   nega_acc/test_count,
				   (float)posi_count/nega_count,
				   (2.0f *(map / test_count) * (mar / test_count)) / ((map / test_count) + (mar / test_count)),
				   map / test_count,
				   mar / test_count,
				   (float)zero_count/test_count
				);
		}
	}
	
	return 0;
}
