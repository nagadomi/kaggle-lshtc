#ifndef NCC_CACHE_HPP
#define NCC_CACHE_HPP
#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>

// Cache for Nearest Centroid Classifier Results
class NCCCache
{
private:
	std::map<int, std::vector<int> > m_cache;
	
public:
	NCCCache(){}
	
	void
	set(unsigned int query_doc_id,
		const std::vector<int> &results)
	{
#ifdef _OPENMP
#pragma omp critical (ncc_cache)
#endif
		{
			m_cache.insert(std::make_pair(query_doc_id, results));
		}
	}
	
	bool
	get(unsigned int query_doc_id, std::vector<int> &results) const
	{
		bool ret = false;
#ifdef _OPENMP
#pragma omp critical (ncc_cache)
#endif
		{
			auto cache = m_cache.find(query_doc_id);
			if (cache != m_cache.end()) {
				results = cache->second;
				ret = true;
			}
		}
		return ret;
	}

	bool
	save(const char *file) const
	{
		FILE *fp = std::fopen(file, "wb");
		if (fp == 0) {
			return false;
		}
		size_t size = m_cache.size();
		std::fwrite(&size, sizeof(size), 1, fp);
		for (auto cache = m_cache.begin(); cache != m_cache.end(); ++cache) {
			int query_doc_id = cache->first;
			size = cache->second.size();
			std::fwrite(&query_doc_id, sizeof(query_doc_id), 1, fp);
			std::fwrite(&size, sizeof(size), 1, fp);
			std::fwrite(cache->second.data(), sizeof(int), size, fp);
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
		m_cache.clear();
		size_t cache_num = 0;
		size_t ret = std::fread(&cache_num, sizeof(cache_num), 1, fp);
		if (ret != 1) {
			std::fprintf(stderr, "NCCCache: %s: invalid format 1\n", file);
			fclose(fp);
			return false;
		}
		for (size_t i = 0; i < cache_num; ++i) {
			int query_doc_id;
			size_t vec_size = 0;
			std::vector<int> results;
			ret = fread(&query_doc_id, sizeof(query_doc_id), 1, fp);
			if (ret != 1) {
				std::fprintf(stderr, "NCCCache: %s: invalid format 2\n", file);
				fclose(fp);
				return false;
			}
			ret = fread(&vec_size, sizeof(vec_size), 1, fp);
			if (ret != 1) {
				std::fprintf(stderr, "NCCCache: %s: invalid format 3\n", file);
				fclose(fp);
				return false;
			}
			int buffer[vec_size];
			ret = fread(&buffer[0], sizeof(int), vec_size, fp);
			if (ret != vec_size) {
				std::fprintf(stderr, "NCCCache: %s: invalid format 4\n", file);
				fclose(fp);
				return false;
			}
			std::copy(buffer, buffer + vec_size, std::back_inserter(results));
			m_cache.insert(std::make_pair(query_doc_id, results));
		}
		fclose(fp);
		
		return true;
	}
};

#endif
