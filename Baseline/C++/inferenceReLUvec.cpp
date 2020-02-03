#include<iostream>
#include<Eigen/Dense>
#include<Eigen/Sparse>

const float YMAX = 32;
Eigen::SparseMatrix<float> inferenceReLUvec(const std::vector<Eigen::SparseMatrix<float> >& W, const std::vector<float>& bias, const Eigen::SparseMatrix<float>& Y0){
	Eigen::SparseMatrix<float> Y(Y0);
	Eigen::SparseMatrix<float> Z(Y.rows(), Y.cols());
	Z.reserve(Y.nonZeros());
	for(size_t i = 0; i < W.size(); ++i){
		Z = (Y * W[i]).pruned();
		Z.coeffs() += bias[i];
		Y = Z.unaryExpr([] (float a) {
				if(a < 0) return 0.0f;
				else if(a > YMAX) return YMAX;
				return a;
				});
	}
	Y.makeCompressed();
	return Y;
} 
