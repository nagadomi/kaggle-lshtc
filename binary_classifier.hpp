#ifndef BINARY_CLASSIFIER
#define BINARY_CLASSIFIER
#include "util.hpp"
#include <cstring>
#include <random>

class BinaryClassifier
{
private:
	typedef std::map<int, float> weight_t;
	typedef std::vector<int> example_index_t;
	typedef std::vector<fv_t> example_t;
	
	weight_t m_w;
	float m_bias;
	
	static inline float
	sigmoid(float x)
	{
		return 1.0f / (1.0f + std::exp(-x));
	}
	inline float
	dot_safe(const fv_t &fv) const
	{
		float dot = 0.0f;
		for (auto x = fv.begin(); x != fv.end(); ++x) {
			auto  w = m_w.find(x->first);
			if (w != m_w.end()) {
				dot += x->second * w->second;
			}
		}
		return dot;
	}
	inline float
	dot(const fv_t &fv) const
	{
		float dot = 0.0f;
		for (auto x = fv.begin(); x != fv.end(); ++x) {
			dot += x->second * m_w.find(x->first)->second;
		}
		return dot;
	}
	void
	reserve(const example_t &examples)
	{
		for (auto fv = examples.begin(); fv != examples.end(); ++fv) {
			for (auto x = fv->begin(); x != fv->end(); ++x) {
				m_w.insert(std::make_pair(x->first, 0.0f));
			}
		}
	}
	inline void
	update(float y, const fv_t &fv, float eta)
	{
		float z = sigmoid(dot(fv) + m_bias);
		for (auto x = fv.begin(); x != fv.end(); ++x) {
			m_w.find(x->first)->second -= eta * (z - y) * x->second;
		}
		m_bias -= eta * (z - y);
	}
	inline float
	uniform(std::mt19937 &rng)
	{
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		return dist(rng);
	}
	inline size_t
	random_index(size_t n, std::mt19937 &rng)
	{
		std::uniform_int_distribution<size_t> dist(0, n - 1);
		return dist(rng);
	}
	void
	initialize(const example_t &posi, const example_t &nega)
	{
		m_w.clear();
		m_bias = 0.0f;
		reserve(posi);
		reserve(nega);
	}

public:
	BinaryClassifier(std::map<int, float> &ws, float bias)
	{
		m_w = ws;
		m_bias = bias;
	}
	BinaryClassifier(const BinaryClassifier &rhs)
	{
		m_w = rhs.m_w;
		m_bias = rhs.bias();
	}
	BinaryClassifier()
	{
	}
	void
	train(const example_t &posi, const example_t &nega,
		  float eta, float p, size_t iteration)
	{
		std::mt19937 rng;
		initialize(posi, nega);
		if (posi.size() == 0) {
			m_bias = -1.0f;
		} else if (nega.size() == 0) {
			m_bias = 1.0f;
		} else {
			size_t count = 0;
			size_t examples = (posi.size() + nega.size());
			for (size_t i = 0; i < iteration; ++i) {
				float learning_rate = eta / (1.0f + (float)i / iteration);
				for (size_t j = 0; j < examples; ++j) {
					if (uniform(rng) < p) {
						size_t k = random_index(posi.size(), rng);
						update(1.0f, posi[k], learning_rate);
					} else {
						size_t k = random_index(nega.size(), rng);
						update(0.0f, nega[k], learning_rate);
					}
					++count;
				}
			}
		}
	}
	size_t
	size(void) const
	{
		size_t nonzero_count = 0;
		for (weight_t::const_iterator w =  m_w.begin(); w != m_w.end(); ++w) {
			if (w->second > 0.0f || w->second < 0.0f) {
				nonzero_count += 1;
			}
		}
		return nonzero_count;
	}
	void
	nonzero_weights(std::map<int, float> &ws) const
	{
		ws.clear();
		for (weight_t::const_iterator w = m_w.begin(); w != m_w.end(); ++w) {
			if (w->second > 0.0f || w->second < 0.0f) {
				ws.insert(std::make_pair(w->first, w->second));
			}
		}
	}
	float
	bias(void) const
	{
		return m_bias;
	}
	float
	predict(const fv_t &fv) const
	{
		return dot_safe(fv) + m_bias;
	}
};

#endif
