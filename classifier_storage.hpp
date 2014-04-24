#ifndef CLASSIFIER_STORAGE_HPP
#define CLASSIFIER_STORAGE_HPP

#include "binary_classifier.hpp"
#include <cstdio>

// Storage for Binary Classifier
class ClassifierStorage
{
private:
	std::map<int, BinaryClassifier> m_classifiers;
	
public:
	ClassifierStorage(){}
	
	void
	set(unsigned int category_id,
		BinaryClassifier &classifier)
	{
#ifdef _OPENMP
#pragma omp critical (classifier_storage)
#endif
		{
			m_classifiers.insert(std::make_pair(category_id, classifier));
		}
	}
	
	const BinaryClassifier *
	get(unsigned int category_id) const
	{
		const BinaryClassifier *classifier = 0;
#ifdef _OPENMP
#pragma omp critical (classifier_storage)
#endif
		{
			auto i = m_classifiers.find(category_id);
			if (i != m_classifiers.end()) {
				classifier = &i->second;
			}
		}
		return classifier;
	}
	bool
	save(const char *file) const
	{
		FILE *fp = std::fopen(file, "wb");
		if (fp == 0) {
			return false;
		}
		size_t size = m_classifiers.size();
		std::fwrite(&size, sizeof(size), 1, fp);
		
		for (auto classifier = m_classifiers.begin();
			 classifier != m_classifiers.end(); ++classifier)
		{
			int category_id = classifier->first;
			std::map<int, float> ws;
			float bias = classifier->second.bias();
			classifier->second.nonzero_weights(ws);
			
			size = ws.size();
			fwrite(&category_id, sizeof(category_id), 1, fp);
			fwrite(&size, sizeof(size), 1, fp);
			//printf("category_id: %d, %ld\n", category_id, size);
			for (auto w = ws.begin(); w != ws.end(); ++w) {
				fwrite(&w->first, sizeof(w->first), 1, fp);
				fwrite(&w->second, sizeof(w->second), 1, fp);
			}
			fwrite(&bias, sizeof(bias), 1, fp);
		}
		fclose(fp);
		
		return true;
	}
	bool
	load(const char *file)
	{
		FILE *fp = std::fopen(file, "rb");
		
		if (fp == 0) {
			return false;
		}
		m_classifiers.clear();
		size_t classifier_num = 0;
		size_t ret = std::fread(&classifier_num, sizeof(classifier_num), 1, fp);
		if (ret != 1) {
			std::fprintf(stderr, "ClassifierStorage: %s: invalid format 1\n", file);
			fclose(fp);
			return false;
		}
		for (size_t i = 0; i < classifier_num; ++i) {
			int category_id;
			size_t vec_size = 0;
			float bias = 0.0f;
			std::map<int, float> ws;
			
			ret = fread(&category_id, sizeof(category_id), 1, fp);
			if (ret != 1) {
				std::fprintf(stderr, "ClassifierStorage: %s: invalid format 2\n", file);
				fclose(fp);
				return false;
			}
			ret = fread(&vec_size, sizeof(vec_size), 1, fp);
			//printf("category_id: %d, %ld\n", category_id, vec_size);
			if (ret != 1) {
				std::fprintf(stderr, "ClassifierStorage: %s: invalid format 3\n", file);
				fclose(fp);
				return false;
			}
			for (size_t i = 0; i < vec_size; ++i) {
				int id;
				float val;
				ret = fread(&id, sizeof(id), 1, fp);
				if (ret != 1) {
					std::fprintf(stderr, "ClassifierStorage: %s: invalid format 4\n", file);
					fclose(fp);
					return false;
				}
				ret = fread(&val, sizeof(val), 1, fp);
				if (ret != 1) {
					std::fprintf(stderr, "ClassifierStorage: %s: invalid format 4\n", file);
					fclose(fp);
					return false;
				}
				ws.insert(std::make_pair(id, val));
			}
			ret = fread(&bias, sizeof(bias), 1, fp);
			if (ret != 1) {
				std::fprintf(stderr, "ClassifierStorage: %s: invalid format 5\n", file);
				fclose(fp);
				return false;
			}
			BinaryClassifier classifier(ws, bias);
			m_classifiers.insert(std::make_pair(category_id, classifier));
		}
		fclose(fp);
		
		return true;
	}
};

#endif
