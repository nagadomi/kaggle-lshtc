#include "util.hpp"
#include "reader.hpp"
#include "tick.hpp"
#include "tfidf_transformer.hpp"
#include "inverted_index.hpp"
#include "evaluation.hpp"
#include <cstdio>
#include <map>
#include "SETTINGS.h"

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
	DataReader reader, test_reader;
	std::vector<fv_t> data;
	std::vector<fv_t> test_data;
	std::vector<label_t> labels;
	std::vector<label_t> dummy_labels;
	category_index_t category_index;
	TFIDFTransformer tfidf;
	long t = tick();
	Evaluation evaluation;
	InvertedIndex knn;
	std::vector<std::pair<int, std::vector<int> > > submission;	
	
	if (!reader.open(TRAIN_DATA)) {
		fprintf(stderr, "open failed: %s\n", TRAIN_DATA);
		return -1;
	}
	if (!test_reader.open(TEST_DATA)) {
		fprintf(stderr, "open failed: %s\n", TEST_DATA);
		return -1;
	}
	reader.read(data, labels);
	test_reader.read(test_data, dummy_labels);
	printf("load: train %ld test %ld %ldms\n",
		   data.size(), test_data.size(), tick() - t);
	reader.close();
	test_reader.close();
	
	t = tick();
	build_category_index(category_index, data, labels);
	
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

