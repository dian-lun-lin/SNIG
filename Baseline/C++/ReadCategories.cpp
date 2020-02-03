#include<iostream>
#include<fstream>
#include<string>
#include<Eigen/Sparse>
#include<sstream>

Eigen::SparseVector<float> ReadCategories(const std::string &S, const size_t &Ncategories){
	std::string line;
	std::istringstream f(S);
	Eigen::SparseVector<float> m(Ncategories);
	m.reserve(Ncategories / 200);
	while(std::getline(f, line)) m.insert(std::stoi(line) - 1) = 1;
	return m;
}
