#include "util.hpp"
#include "reader.hpp"
#include "tick.hpp"
#include "tfidf_transformer.hpp"
#include "nearest_centroid_classifier.hpp"
#include "evaluation.hpp"
#include <cstdio>
#include <map>
#include "SETTINGS.h"

// simple Centroid Classifier baseline

#define K       4

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

int
main(void)
{
	DataReader reader;
	std::vector<fv_t> data;
	std::vector<fv_t> test_data;
	std::vector<label_t> labels;
	std::vector<label_t> test_labels;
	category_index_t category_index;
	NearestCentroidClassifier centroid_classifier;
	TFIDFTransformer tfidf;
	long t = tick();
	long t_all = tick();
	Evaluation evaluation;
	
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
	centroid_classifier.train(category_index, data);
	printf("build index %ldms\n", tick() -t );
	
	t = tick();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)	
#endif
	for (int i = 0; i < (int)test_data.size(); ++i) {
		std::vector<int> topn_labels;
		centroid_classifier.predict(topn_labels, K, test_data[i]);
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
