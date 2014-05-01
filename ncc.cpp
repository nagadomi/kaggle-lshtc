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

void
make_submission(const std::vector<std::pair<int, std::vector<int> > > &submission)
{
	FILE *fp = fopen(SUBMISSION, "w");
	fprintf(fp, "Id,Predicted\n");
	for (auto i = submission.begin(); i != submission.end(); ++i) {
		bool first = true;
		fprintf(fp, "%d,", i->first + 1);
		for (auto j = i->second.begin(); j != i->second.end(); ++j)	{
			if (first) {
				first = false;
			} else {
				fprintf(fp, " ");
			}
			fprintf(fp, "%d", *j);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

int
main(void)
{
	DataReader reader, test_reader;
	std::vector<fv_t> data;
	std::vector<fv_t> test_data;
	std::vector<label_t> labels;
	std::vector<label_t> dummy_labels;
	category_index_t category_index;
	NearestCentroidClassifier centroid_classifier;
	TFIDFTransformer tfidf;
	long t = tick();
	std::vector<std::pair<int, std::vector<int> > > submission;	
	
	if (!reader.open(TRAIN_DATA)) {
		fprintf(stderr, "cant read file\n");
		return -1;
	}
	if (!test_reader.open(TEST_DATA)) {
		fprintf(stderr, "open failed: %s\n", TEST_DATA);
		return -1;
	}
	reader.read(data, labels);
	test_reader.read(test_data, dummy_labels);
	printf("load train %ld, test %ld, %ldms\n",
		   data.size(), test_data.size(), tick() - t);
	reader.close();
	test_reader.close();	
	
	t = tick();
	build_category_index(category_index, data, labels);
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
			submission.push_back(std::make_pair(i, topn_labels));
			if (i % 1000 == 0) {
				printf("--- predict %d/%ld %ldms\n", i, test_data.size(), tick() -t);
				t = tick();
			}
		}
	}
	std::sort(submission.begin(), submission.end());
	make_submission(submission);
	
	return 0;
}
