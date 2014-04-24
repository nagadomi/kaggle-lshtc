#include "util.hpp"
#include "reader.hpp"
#include "tick.hpp"
#include "tfidf_transformer.hpp"
#include "evaluation.hpp"
#include "classifier_storage.hpp"
#include "nearest_centroid_classifier.hpp"
#include <cstdio>
#include "SETTINGS.h"

static void
predict_labels(std::vector<int> &results,
			   const fv_t &query,
			   const std::vector<int> &search_results,
			   ClassifierStorage &classifiers)
{
	std::vector<int> candidate_labels;
	std::vector<std::pair<double, int> > rank;
	
	for (auto doc = search_results.begin(); doc != search_results.end(); ++doc) {
		candidate_labels.push_back(*doc);
	}
	for (int i = 0; i < (int)candidate_labels.size(); ++i) {
		const BinaryClassifier *classifier = classifiers.get(candidate_labels[i]);
		if (classifier != 0) {
			float value = classifier->predict(query);
			rank.push_back(std::make_pair(value, candidate_labels[i]));
		}
	}
	std::sort(rank.begin(), rank.end(),
			  std::greater<std::pair<double, int> >());
	for (auto i = rank.begin(); i != rank.end(); ++i) {
		if (results.size() == 0 || i->first >= 0.0) {
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
	ClassifierStorage classifier_storage;
	NearestCentroidClassifier centroid;
	TFIDFTransformer transformer;
	long t = tick();
	long t_all = tick();	
	Evaluation evaluation;

	if (!reader.open(TRAIN_DATA)) {
		fprintf(stderr, "cant read file\n");
		return -1;
	}
	if (!classifier_storage.load(MODEL)) {
		fprintf(stderr, "cant open classifier storage\n");
		return -1;
	}
	reader.read(data, labels);
	reader.close();
	
	printf("read %ld, %ld, %ldms\n", data.size(), labels.size(), tick() - t);
	
	t = tick();
	build_category_index(category_index, data, labels);
	srand(VT_SEED);
	split_data(test_data, test_labels, data, labels, category_index, 0.05f);
	build_category_index(category_index, data, labels);
	printf("split train:%ld, test:%ld\n", data.size(), test_data.size());
	
	t = tick();
	transformer.load(WEIGHT);
	transformer.transform(data);
	transformer.transform(test_data);
	centroid.load(CENTROID);
	printf("build index %ldms\n", tick() -t );
	
	t = tick();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)	
#endif
	for (int i = 0; i < (int)test_data.size(); ++i) {
		std::vector<int> topn_labels;
		std::vector<int> results;		
		centroid.predict(results, K_PREDICT, test_data[i]);
		predict_labels(topn_labels, test_data[i], results, classifier_storage);
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
