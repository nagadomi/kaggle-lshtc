#ifndef LSHTC4_EVALUATION_HPP
#define LSHTC4_EVALUATION_HPP
#include "util.hpp"

class Evaluation
{
private:
	size_t m_example_count;
	size_t m_top_true_positive;
	std::vector<int> m_true_positive;
	std::vector<int> m_false_negative;
	std::vector<int> m_false_positive;
	label_t m_labels;

	void
	resize_all(unsigned int label_id)
	{
		m_labels.insert(label_id);
		if (label_id >= m_true_positive.size()) {
			m_true_positive.resize(label_id + 1);
		}
		if (label_id >= m_false_negative.size()) {
			m_false_negative.resize(label_id + 1);
		}
		if (label_id >= m_false_positive.size()) {
			m_false_positive.resize(label_id + 1);
		}
	}

public:
	Evaluation() : m_example_count(0) {}
	
	void
	clear()
	{
		m_example_count = 0;
		m_top_true_positive = 0;
		m_true_positive.clear();
		m_false_negative.clear();
		m_false_positive.clear();
		m_labels.clear();
	}
	void
	update(const std::vector<int> &label_prediction,
		   const label_t &label_actual)
	{
		label_t label_pre;
		bool top = true;
		
		m_example_count += 1;
		for (std::vector<int>::const_iterator label = label_prediction.begin();
			 label != label_prediction.end();
			 ++label)
		{
			resize_all(*label);
			label_pre.insert(*label);
			if (label_actual.find(*label) != label_actual.end()) {
				m_true_positive[*label] += 1;
				if (top) {
					m_top_true_positive += 1;
				}
			} else {
				m_false_positive[*label] += 1;
			}
			if (top) {
				top = false;
			}
		}
		for (label_t::const_iterator label = label_actual.begin();
			 label != label_actual.end();
			 ++label)
		{
			resize_all(*label);
			if (label_pre.find(*label) == label_pre.end()) {
				m_false_negative[*label] += 1;
			}
		}
	}
	void
	score(double &maf, double &map, double &mar, double &top1_accuracy) const
	{
		double label_count = (double)m_labels.size();
		
		top1_accuracy = (double)m_top_true_positive / m_example_count;
		
		maf = map = mar = 0.0;
		for (size_t i = 0; i < m_true_positive.size(); ++i) {
			double tp = m_true_positive[i];
			double tp_fp = tp + m_false_positive[i];
			double tp_fn = tp + m_false_negative[i];
			if (tp_fp > 0.0 && tp_fn > 0.0) {
				map += tp / (tp_fp);
				mar += tp / (tp_fn);
			}
		}
		map /= label_count;
		mar /= label_count;
		maf = (2.0 * map * mar) / (map + mar);
	}
};

#endif
