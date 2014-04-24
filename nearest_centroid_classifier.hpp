#ifndef NEAREST_CENTROID_CLASSIFIER_HPP
#define NEAREST_CENTROID_CLASSIFIER_HPP
#include "util.hpp"
#include "inverted_index.hpp"

class NearestCentroidClassifier
{
private:
	std::vector<fv_t> m_centroids;
	std::vector<int> m_centroid_labels;
	InvertedIndex m_inverted_index;
	
	static void
	vector_sum(fv_t &sum,
			   const std::vector<int> &indexes,
			   const std::vector<fv_t> &data)
	{
		sum.clear();
		for (auto i = indexes.begin(); i != indexes.end(); ++i) {
			const fv_t &x = data[*i];
			for (auto word = x.begin(); word != x.end(); ++word) {
				auto s = sum.find(word->first);
				if (s != sum.end()) {
					s->second += word->second;
				} else {
					sum.insert(std::make_pair(word->first, word->second));
				}
			}
		}
	}
	
	static void
	vector_normalize_l2(fv_t &x)
	{
		double dot = 0.0f;
		for (auto i = x.begin(); i != x.end(); ++i) {
			dot += i->second * i->second;
		}
		if (dot > 0.0f) {
			double scale = 1.0f / std::sqrt(dot);
			for (auto i = x.begin(); i != x.end(); ++i) {
				i->second *= scale;
			}
		}
	}
	
public:
	NearestCentroidClassifier(){}
	
	void
	train(const category_index_t &category_index,
		  const std::vector<fv_t> &data)
	{
		for (auto l = category_index.begin(); l != category_index.end(); ++l) {
			fv_t centroid;
			vector_sum(centroid, l->second, data);
			vector_normalize_l2(centroid);
			m_centroids.push_back(centroid);
			m_centroid_labels.push_back(l->first);
		}
		m_inverted_index.build(&m_centroids);
	}
	
	inline void
	predict(std::vector<int> &results,
			size_t k,
			const fv_t &query) const
	{
		InvertedIndex::result_t knn;
		
		m_inverted_index.knn(knn, k, query);
		results.clear();
		for (auto i = knn.begin(); i != knn.end(); ++i) {
			results.push_back(m_centroid_labels[i->id]);
		}
	}

	size_t
	size(void) const
	{
		return m_centroids.size();
	}

	bool
	save(const char *file) const
	{
		FILE *fp = std::fopen(file, "wb");
		if (fp == 0) {
			return false;
		}
		size_t size = m_centroids.size();
		std::fwrite(&size, sizeof(size), 1, fp);
		for (auto centroid = m_centroids.begin();
			 centroid != m_centroids.end(); ++centroid)
		{
			size = centroid->size();
			std::fwrite(&size, sizeof(size), 1, fp);
			for (auto w = centroid->begin(); w != centroid->end(); ++w) {
				std::fwrite(&w->first, sizeof(w->first), 1, fp);
				std::fwrite(&w->second, sizeof(w->second), 1, fp);
			}
		}
		size = m_centroid_labels.size();
		std::fwrite(&size, sizeof(size), 1, fp);
		std::fwrite(m_centroid_labels.data(), sizeof(int), size, fp);
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
		m_centroids.clear();
		m_centroid_labels.clear();
		m_inverted_index.clear();
		
		size_t centroid_num = 0;
		size_t ret = std::fread(&centroid_num, sizeof(centroid_num), 1, fp);
		if (ret != 1) {
			std::fprintf(stderr, "%s: invalid format 1\n", file);
			fclose(fp);
			return false;
		}
		for (size_t i = 0; i < centroid_num; ++i) {
			fv_t centroid;
			size_t word_num = 0;
			ret = fread(&word_num, sizeof(word_num), 1, fp);
			if (ret != 1) {
				std::fprintf(stderr, "%s: invalid format 2\n", file);
				fclose(fp);
				return false;
			}
			for (size_t j = 0; j < word_num; ++j) {
				int word_id;
				float word_weight;
				ret = std::fread(&word_id, sizeof(word_id), 1, fp);
				if (ret != 1) {
					std::fprintf(stderr, "%s: invalid format 3\n", file);
					fclose(fp);
					return false;
				}
				ret = std::fread(&word_weight, sizeof(word_weight), 1, fp);
				if (ret != 1) {
					std::fprintf(stderr, "%s: invalid format 4\n", file);
					fclose(fp);
					return false;
				}
				centroid.insert(std::make_pair(word_id, word_weight));
			}
			m_centroids.push_back(centroid);
		}
		ret = std::fread(&centroid_num, sizeof(centroid_num), 1, fp);
		if (ret != 1) {
			std::fprintf(stderr, "%s: invalid format 5\n", file);
			fclose(fp);
			return false;
		}
		int *buffer = new int[centroid_num];
		ret = std::fread(buffer, sizeof(int), centroid_num, fp);
		if (ret != centroid_num) {
			std::fprintf(stderr, "%s: invalid format 6\n", file);
			delete buffer;
			fclose(fp);
			return false;
		}
		std::copy(buffer, buffer + centroid_num,
				  std::back_inserter(m_centroid_labels));
		delete buffer;
		
		fclose(fp);
		
		m_inverted_index.build(&m_centroids);
		
		return true;
	}
};

#endif
