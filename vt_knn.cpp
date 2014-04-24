#include "util.hpp"
#include "reader.hpp"
#include "tick.hpp"
#include "tfidf_transformer.hpp"
#include "inverted_index.hpp"
#include "evaluation.hpp"
#include <cstdio>
#include <map>
#include "SETTINGS.h"

// simple k-NN baseline

#define K              12
#define K_FIRST        3000
#define PREDICT_LABEL  5

static void
predict(std::vector<int> &results,
		const InvertedIndex::result_t &search_results,
		const std::vector<label_t> &labels)
{
	std::map<int, float> score;
	std::vector<std::pair<float, int> > tmp;
	
	results.clear();
	
	for (auto i = search_results.begin(); i != search_results.end(); ++i) {
		const label_t &label = labels[i->id];
		for (auto j = label.begin(); j != label.end(); ++j)	{
			auto s = score.find(*j);
			if (s != score.end()) {
				s->second += 1.0f + i->cosine * 0.1f;
			} else {
				score.insert(std::make_pair(*j, 1.0f + i->cosine * 0.1f));
			}
		}
	}
	
	for (auto i = score.begin(); i != score.end(); ++i)	{
		tmp.push_back(std::make_pair(i->second, i->first));
	}
	std::sort(tmp.begin(), tmp.end(), std::greater<std::pair<float, int> >());
	for (auto i = tmp.begin(); i != tmp.end(); ++i) {
		if (results.size() < PREDICT_LABEL) {
			results.push_back(i->second);
		}
	}
}

static void
print_evaluation(const Evaluation &evaluation, int i, long t)
{
	double maf, map, mar, top1_acc;
	evaluation.score(maf, map, mar, top1_acc);
	
	printf("--- %d MaF: %f, MaP:%f, MaR:%f, Top1ACC: %f %ldms\n",
		   i,
		   maf, map, mar, top1_acc,
		   tick() -t);
}

int main(void)
{
	DataReader reader;
	std::vector<fv_t> data;
	std::vector<fv_t> test_data;
	std::vector<label_t> labels;
	std::vector<label_t> test_labels;
	category_index_t category_index;
	TFIDFTransformer tfidf;
	long t = tick();
	long t_all = tick();
	Evaluation evaluation;
	InvertedIndex knn;
	
	if (!reader.open(TRAIN_DATA)) {
		fprintf(stderr, "cant read file\n");
		return -1;
	}
	reader.read(data, labels);
	printf("read %ld, %ld, %ldms\n", data.size(), labels.size(), tick() - t);
	reader.close();
	
	t = tick();
	srand(VT_SEED);
	build_category_index(category_index, data, labels);
	split_data(test_data, test_labels, data, labels, category_index, 0.05f);
	build_category_index(category_index, data, labels);
	printf("split train:%ld, test:%ld\n", data.size(), test_data.size());
	
	t = tick();
	tfidf.train(data);
	tfidf.transform(data);
	tfidf.transform(test_data);
	knn.build(&data);
	printf("build index %ldms\n", tick() -t );
	
	t = tick();
	
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
	for (int i = 0; i < (int)test_data.size(); ++i) {
		std::vector<int> topn_labels;
		InvertedIndex::result_t results;
		
		knn.fast_knn(results, K, test_data[i], K_FIRST, data.size() / 100);
		predict(topn_labels, results, labels);
		
#ifdef _OPENMP
#pragma omp critical
#endif
		{
			evaluation.update(topn_labels, test_labels[i]);
			if (i % 1000 == 0) {
				print_evaluation(evaluation, i, t);
				t = tick();
			}
		}
	}
	printf("----\n");
	print_evaluation(evaluation, test_data.size(), t_all);
	
	return 0;
}
