#ifndef INVERTED_INDEX_HPP
#define INVERTED_INDEX_HPP

#include "util.hpp"

// Inverted Index for k-NN
class InvertedIndex
{
public:
	typedef struct result {
		int id;
		float cosine;
		
		result(int id, float cosine)
		{
			this->id = id;
			this->cosine = cosine;
		}
		inline bool
		operator>(const struct result &rhs) const
		{
			return cosine > rhs.cosine;
		}
	} result_element_t;
	typedef std::vector<result_element_t> result_t;
	
private:
	typedef struct inverted_index_word {
		int doc_id;
		float value;
		
		inverted_index_word(unsigned int doc_id, float value)
		{
			this->doc_id = doc_id;
			this->value = value;
		}
		inline bool
		operator<(const struct inverted_index_word &rhs) const
		{
			return doc_id < rhs.doc_id;
		}
	} inverted_index_word_t;
	typedef std::vector<inverted_index_word_t> inverted_index_doc_t;
	typedef std::vector<inverted_index_doc_t> inverted_index_t;
	
	inverted_index_t m_inverted_index;
	const std::vector<fv_t> *m_data;
	
	typedef struct word_result {
		int id;
		float dot;
		
		word_result(int id, float dot)
		{
			this->id = id;
			this->dot = dot;
		}
		inline bool
		operator<(const struct word_result &rhs) const
		{
			return id < rhs.id;
		}
	} word_result_t;
	
	// min heap
	typedef std::priority_queue<result_element_t,
								std::vector<result_element_t>,
								std::greater<std::vector<result_element_t>::value_type> > topn_t;
	
	static inline void
	topn_push(topn_t &topn, size_t k, int id, float cosine)
	{
		if (k > topn.size()) {
			topn.push(result_element_t(id, cosine));
		} else if (topn.top().cosine < cosine) {
			topn.push(result_element_t(id, cosine));
			topn.pop();
		}
	}
	
	static inline void
	topn_convert(result_t &results, size_t k, topn_t &topn)
	{
		results.clear();
		while (topn.size() > 0) {
			results.push_back(topn.top());
			topn.pop();
		}
		std::reverse(results.begin(), results.end());
		if (results.size() > k) {
			results.erase(results.begin() + k, results.end());
		}
	}
	
	static inline float
	fv_cosine(const fv_t &fv1, const fv_t &fv2)
	{
		float dot = 0.0f;
		for (auto i = fv1.begin(); i != fv1.end(); ++i) {
			auto j = fv2.find(i->first);
			if (j != fv2.end()) {
				dot += 2.0f * i->second * j->second;
			}
		}
		return dot;
	}
	
	fv_t
	truncate_query(const fv_t &fv, size_t threshold) const
	{
		fv_t ret;
		for (auto word = fv.begin(); word != fv.end(); ++word) {
			if (word->first < (int)m_inverted_index.size()
				&& m_inverted_index[word->first].size() < threshold)
			{
				ret.insert(std::make_pair(word->first, word->second));
			}
		}
		
		return ret;
	}

public:
	InvertedIndex() {}
	
	void
	build(const std::vector<fv_t> *data)
	{
		m_data = data;
		
		clear();
		for (size_t id = 0; id < m_data->size(); ++id) {
			this->set(id, m_data->at(id));
		}
	}
	void
	clear()
	{
		m_inverted_index.clear();
	}
	void
	set(unsigned int id, const fv_t &fv)
	{
		for (fv_t::const_iterator word = fv.begin(); word != fv.end(); ++word) {
			if (word->first >= (int)m_inverted_index.size()) {
				m_inverted_index.resize(word->first + 1);
			}
			m_inverted_index[word->first].push_back(inverted_index_word_t(id, word->second));
		}
	}
	void
	fast_knn(result_t &results,
			 size_t k,
			 const fv_t &fv,
			 size_t first_k,
			 size_t first_truncate_threshold) const
	{
		// knn using few features
		fv_t query = fv;
		fv_t query_first = truncate_query(query, first_truncate_threshold);
		this->knn(results, first_k, query_first);
		if (results.size() == 0) {
			this->knn(results, k, query);
		}
		
		/* knn using full features */
		size_t threads = processor_count();
		topn_t topn[threads];
		
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
		for (int i = 0; i < (int)results.size(); ++i) {
			size_t thread_id = processor_id();
			topn_push(topn[thread_id], k,
					  results[i].id,
					  fv_cosine(query, m_data->at(results[i].id)));
		}
		for (size_t i = 1; i < threads; ++i) {
			while (!topn[i].empty()) {
				const result_element_t &elm = topn[i].top();
				topn_push(topn[0], k, elm.id, elm.cosine);
				topn[i].pop();
			}
		}
		results.clear();
		topn_convert(results, k, topn[0]);
	}
	
	void
	knn(result_t &results,
		size_t k,
		const fv_t &query) const
	{
		int num_threads = processor_count();
		std::vector<std::vector<word_result_t> > hits;
		std::vector<std::pair<unsigned int, float> > fv;
		topn_t topn;
		
		for (fv_t::const_iterator i = query.begin();
			 i != query.end();
			 ++i)
		{
			fv.push_back(std::make_pair(i->first, i->second));
		}
		
		hits.resize(num_threads);
		hits[0].reserve(m_data->size() / 4 + 1);
		for (int i = 1; i < num_threads; ++i) {
			hits[i].reserve(m_data->size() / 4 / num_threads + 1);
		}
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 4)
#endif
		for (int i = 0; i < (int)fv.size(); ++i) {
			int thread_id = processor_id();
			float query_w = 2.0f * fv[i].second;
			
			std::vector<word_result_t> &hit = hits[thread_id];
			if (fv[i].first < m_inverted_index.size()) {
				const inverted_index_doc_t &ids = m_inverted_index[fv[i].first];
				for (inverted_index_doc_t::const_iterator doc = ids.begin();
					 doc != ids.end();
					 ++doc)
				{
					hit.push_back(word_result_t(doc->doc_id, query_w * doc->value));
				}
			}
		}
		for (int i = 1; i < num_threads; ++i) {
			std::copy(hits[i].begin(), hits[i].end(), std::back_inserter(hits[0]));
		}
		std::sort(hits[0].begin(), hits[0].end());
		if (hits[0].size() > 0) {
			int id = hits[0][0].id;
			float cosine = 0.0f;
			std::vector<word_result_t> &hit = hits[0];
			for (auto j = hit.begin(); j != hit.end(); ++j)	{
				if (j->id == id) {
					cosine += j->dot;
				} else {
					topn_push(topn, k, id, cosine);
					id = j->id;
					cosine = j->dot;
				}
			}
			topn_push(topn, k, id, cosine);
		}
		topn_convert(results, k, topn);
	}
};

#endif
