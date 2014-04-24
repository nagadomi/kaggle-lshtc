#ifndef READER_H
#define READER_H

#include "util.hpp"
#include <fstream>
#include <sstream>

class DataReader
{
private:
	std::ifstream m_fp;
	char m_buffer[1024 * 1024];
public:
	bool
	open(const char *file)
	{
		m_fp.open(file, std::ifstream::in);
		if (!m_fp) {
			return false;
		}
		m_fp.rdbuf()->pubsetbuf(m_buffer, sizeof(m_buffer));
		
		return true;
	}
	
	void
	read(std::vector<fv_t> &data,
		 std::vector<label_t> &labels)
	{
		std::string line;
		data.clear();
		labels.clear();
		
		getline(m_fp, line); // skip headeer
		while (getline(m_fp, line)) {
			std::istringstream is(line);
			fv_t fv;
			label_t label;
			char sep;
			float value;
			int id;

			is >> std::noskipws;
			while (is >> id >> sep) {
				if (sep == ',') {
					label.insert(id);
					is >> sep;
					if (sep != ' ') {
						is.putback(sep);
					}
				} else {
					label.insert(id);
					break;
				}
			}
			while (is >> id >> sep >> value) {
				fv.insert(std::make_pair(id, value));
				is >> sep;
			}
			data.push_back(fv);
			labels.push_back(label);
		}
	}
	
	void
	close(void)
	{
		m_fp.close();
	}
};

#endif
