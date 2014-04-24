#include "util.hpp"
#include "reader.hpp"
#include "tick.hpp"
#include "nearest_centroid_classifier.hpp"
#include "tfidf_transformer.hpp"
#include "classifier_storage.hpp"
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

bool
read_data(std::vector<fv_t> &data,
		  std::vector<label_t> &labels,
		  std::vector<fv_t> &test_data)
{
	DataReader reader;
	DataReader test_reader;
	std::vector<label_t> *dummy_labels = new std::vector<label_t>;
	
	if (!reader.open(TRAIN_DATA)) {
		fprintf(stderr, "open failed: %s:\n", TRAIN_DATA);
		return false;
	}
	if (!test_reader.open(TEST_DATA)) {
		fprintf(stderr, "open failed: %s\n", TEST_DATA);
		return false;
	}
	reader.read(data, labels);
	test_reader.read(test_data, *dummy_labels);
	
	reader.close();
	test_reader.close();
	delete dummy_labels;

	return true;
}

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

int main(void)
{
	std::vector<fv_t> test_data;
	std::vector<label_t> test_labels;
	TFIDFTransformer transformer;
	NearestCentroidClassifier centroid;	
	ClassifierStorage classifier_storage;
	std::vector<std::pair<int, std::vector<int> > > submission;
	long t = tick();
	DataReader reader;
	
	if (!classifier_storage.load(MODEL)) {
		fprintf(stderr, "cant open classifier storage\n");
		return -1;
	}
	if (!reader.open(TEST_DATA)) {
		fprintf(stderr, "open failed: %s\n", TEST_DATA);
		return -1;
	}
	reader.read(test_data, test_labels);
	printf("read test: %ld, %ldms\n",
		   test_data.size(), tick() - t);
	t = tick();
	
	transformer.load(WEIGHT);
	transformer.transform(test_data);
	centroid.load(CENTROID);
	
	t = tick();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
	for (int id = 0; id < (int)test_data.size(); ++id) {
		std::vector<int> topn_labels;
		std::vector<int> results;
		
		centroid.predict(results, K_PREDICT, test_data[id]);
		predict_labels(topn_labels, test_data[id], results, classifier_storage);
		
#ifdef _OPENMP
#pragma omp critical (submission)
#endif
		{
			submission.push_back(std::make_pair(id, topn_labels));
			if (id % 10000 == 0) {
				printf("--- predict %d/%ld %ldms\n", id, test_data.size(), tick() -t);
				t = tick();
			}
		}
	}
	std::sort(submission.begin(), submission.end());
	make_submission(submission);
	
	return 0;
}
