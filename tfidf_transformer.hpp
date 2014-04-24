#ifndef TFIDF_TRANSFORMER_HPP
#define TFIDF_TRANSFORMER_HPP
#include "util.hpp"
#include <cstdio>

class TFIDFTransformer
{
private:
	typedef std::vector<float> idf_t;
	idf_t m_idf;
	float m_zero_idf;
	
public:
	TFIDFTransformer(){}
	
	void
	train(const std::vector<fv_t> &tf)
	{
		static const float BETA = 5.0f;
		double docs = (double)tf.size();
		
		// word count
		m_idf.clear();
		for (auto doc = tf.begin(); doc != tf.end(); ++doc)	{
			for (auto word = doc->begin(); word != doc->end(); ++word) {
				if (word->first >= (int)m_idf.size()) {
					m_idf.resize(word->first + 1);
				}
				m_idf[word->first] += 1.0f;
			}
		}
		// compute idf
		for (auto idf = m_idf.begin(); idf != m_idf.end(); ++idf) {
			*idf = BETA + std::log(docs / (*idf + 1.0f));
		}
		m_zero_idf = BETA + std::log(docs);
	}

	void
	transform(fv_t &fv) const
	{
		float dot = 0.0f;
		for (auto word = fv.begin(); word != fv.end(); ++word) {
			float idf;
			float tf = std::log(word->second + 1.0);// + 1.0;
			if (word->first < (int)m_idf.size()) {
				idf = m_idf[word->first];
			} else {
				idf = m_zero_idf;
			}
			word->second = tf * idf;
			dot += word->second * word->second;
		}
		if (dot > 0.0f) {
			// L2 normalize
			float norm_scale = 1.0f / std::sqrt(dot);
			for (auto word = fv.begin(); word != fv.end(); ++word) {
				word->second *= norm_scale;
			}
		}
	}
	
	void
	transform(std::vector<fv_t> &tf) const
	{
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (size_t i = 0; i < tf.size(); ++i) {
			transform(tf[i]);
		}
	}

	bool
	save(const char *file) const
	{
		FILE *fp = std::fopen(file, "wb");
		if (fp == 0) {
			return false;
		}
		size_t size = m_idf.size();
		std::fwrite(&size, sizeof(size), 1, fp);
		std::fwrite(m_idf.data(), sizeof(float), size, fp);
		std::fwrite(&m_zero_idf, sizeof(m_zero_idf), 1, fp);
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
		size_t size = 0;
		size_t ret = std::fread(&size, sizeof(size), 1, fp);
		if (ret != 1) {
			std::fprintf(stderr, "%s: invalid format 1\n", file);
			fclose(fp);
			return false;
		}
		m_idf.clear();
		
		float *buffer = new float[size];
		ret = fread(&buffer[0], sizeof(float), size, fp);
		if (ret != size) {
			std::fprintf(stderr, "%s: invalid format 2\n", file);
			delete buffer;
			fclose(fp);
			return false;
		}
		std::copy(buffer, buffer + size, std::back_inserter(m_idf));
		delete buffer;
		ret = std::fread(&m_zero_idf, sizeof(m_zero_idf), 1, fp);
		if (ret != 1) {
			std::fprintf(stderr, "%s: invalid format 2\n", file);
			fclose(fp);
			return false;
		}
		fclose(fp);
		
		return true;
	}
};

#endif
