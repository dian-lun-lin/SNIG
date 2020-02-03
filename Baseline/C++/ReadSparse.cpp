#include<iostream>
#include<fstream>
#include<string>
#include<Eigen/Sparse>

Eigen::SparseMatrix<float> ReadSparse(const std::string &S, const size_t &nNeurons, const size_t &nInput, const bool &isInput){
	typedef Eigen::Triplet<float> T;
	std::string line;
	std::vector<T> tripletList;
	Eigen::SparseMatrix<float> mat;
	if(isInput){
		mat.conservativeResize(nInput, nNeurons);
		tripletList.reserve((nInput*nNeurons)/200);
	} 
	else {
		mat.conservativeResize(nNeurons, nNeurons);
		tripletList.reserve((nNeurons^2)/200);
	}
	std::istringstream f(S);
	while(std::getline(f, line)){
		std::istringstream lineStream(line);
		std::string token;
		std::vector<float> tokens;
		while(std::getline(lineStream, token, '\t'))	tokens.push_back(std::stof(token));
		tripletList.push_back(T(tokens[0] - 1, tokens[1] - 1, tokens[2]));
	}
	mat.reserve(tripletList.size());
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	mat.makeCompressed();
	return mat;
}
