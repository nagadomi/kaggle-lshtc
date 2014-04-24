#ifndef UTIL_H
#define UTIL_H
#include <map>
#include <vector>
#include <string>
#include <set>
#include <queue>
#include <cmath>
#include <algorithm>
#include <functional>
#include <cfloat>

#ifdef _OPENMP
#  include <omp.h>
#endif

typedef std::map<int, float> fv_t;
typedef std::set<int> label_t;
typedef std::map<int, std::vector<int> > category_index_t;

static inline int
processor_count(void)
{
#ifdef _OPENMP
	return omp_get_num_procs();
#else
	return 1;
#endif
}

static inline int
processor_id(void)
{
#ifdef _OPENMP
	return omp_get_thread_num();
#else
	return 0;
#endif
}

static void
build_category_index(category_index_t &index,
					 const std::vector<fv_t> &data,
					 const std::vector<label_t> &labels)
{
	index.clear();
	for (size_t id = 0; id < data.size(); ++id) {
		label_t label = labels[id];
		for (auto i = label.begin(); i != label.end(); ++i)	{
			auto l = index.find(*i);
			if (l == index.end()) {
				std::vector<int> docs;
				docs.push_back(id);
				index.insert(std::make_pair(*i, docs));
			} else {
				l->second.push_back(id);
			}
		}
	}
}

static inline double
rand01(void)
{
	return ((double)rand() / (RAND_MAX));
}

static inline size_t
rand_index(size_t n)
{
	size_t i = n * rand01();
	if (i >= n) {
		i = n - 1;
	}
	return i;
}

static void
shuffle_data(std::vector<fv_t> &data,
			 std::vector<label_t> &labels)
{
	for (size_t i = 0; i < data.size(); ++i) {
		size_t rand_i = rand_index(data.size());
		fv_t tmp1;
		tmp1 = data[i];
		data[i] = data[rand_i];
		data[rand_i] = tmp1;
		
		label_t tmp2;
		std::copy(labels[i].begin(), labels[i].end(),
				  std::inserter(tmp2, tmp2.begin()));
		labels[i].clear();
		std::copy(labels[rand_i].begin(), labels[rand_i].end(),
				  std::inserter(labels[i], labels[i].begin()));
		labels[rand_i].clear();
		std::copy(tmp2.begin(), tmp2.end(),
				  std::inserter(labels[rand_i], labels[rand_i].begin()));
	}
}

void
split_data(std::vector<fv_t> &test_data,
		   std::vector<label_t> &test_labels,
		   std::vector<fv_t> &data,
		   std::vector<label_t> &labels,
		   const category_index_t &category_index,
		   float test_ratio)
{
	std::set<int> test;
	std::vector<fv_t> train_data;
	std::vector<label_t> train_labels;
	
	for (auto i = category_index.begin(); i != category_index.end(); ++i) {
		for (auto j = i->second.begin(); j != i->second.end(); ++j)	{
			if (rand01() < test_ratio) {
				test.insert(*j);
			}
		}
	}
	test_data.clear();
	test_labels.clear();
	test_data.reserve(test.size());
	test_labels.reserve(test.size());
	train_data.reserve(data.size() - test.size());
	train_labels.reserve(data.size() - test.size());
	
	for (size_t i = 0; i < data.size(); ++i) {
		if (test.find(i) != test.end()) {
			test_data.push_back(data[i]);
			test_labels.push_back(labels[i]);
		} else {
			train_data.push_back(data[i]);
			train_labels.push_back(labels[i]);
		}
	}
	data = train_data;
	labels = train_labels;

	shuffle_data(data, labels);
	shuffle_data(test_data, test_labels);
}

#endif
