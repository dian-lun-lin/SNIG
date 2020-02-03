#include<iostream>
#include<string>
#include <chrono>
#include<Eigen/Sparse>
#include "./ReadCategories.cpp"
#include "./ReadSparse.cpp"
#include "./inferenceReLUvec.cpp"
#include "./StrFileRead.cpp"

const std::string INPUTFILE = "../MNIST/sparse-images-";
const std::string CATEGORYFILE = "../MNIST/neuron";
const std::string LAYERFILE = "../neuron";

//currently ignore SAVCAT and READMAT.
const bool SAVECAT = false;
const bool READTSV = true;
const bool READMAT = false;

//total num of MNIST input
const int numMNIST = 60000;

//Select DNN to run.
//Nneuron = [1024, 4096, 16384, 65536];
int Nneuron[] = {1024};

//Select number of layers to run.
//maxLayers = 120 * [1, 4, 16];
int maxLayers[] = {120};

//Set DNN bias
float neuralNetBias[] = {-0.3f, -0.35f, -0.4f, -0.45f};

int main(){
	size_t Nlen = sizeof(Nneuron)/sizeof(Nneuron[0]);
	size_t maxLayersLen = sizeof(maxLayers)/sizeof(maxLayers[0]);
	Eigen::SparseMatrix<float> featureVectors(numMNIST, 1);
	featureVectors.reserve(numMNIST/200);
	for(size_t i = 0; i < Nlen; ++i){
		if(READTSV) featureVectors = ReadSparse(StrFileRead(INPUTFILE + std::to_string(Nneuron[i]) + ".tsv"), Nneuron[i], numMNIST, true);
		//else if(READMAT)
		int NfeatureVectors = featureVectors.cols();

		//Read layers.
		for(size_t j = 0; j < maxLayersLen; ++j){
			Eigen::SparseVector<float> trueCategories;
			if (!SAVECAT) trueCategories = ReadCategories(StrFileRead(CATEGORYFILE + std::to_string(Nneuron[i]) + "-l" + std::to_string(maxLayers[j]) +  "-categories.tsv"), numMNIST);
			int DNNedges = 0;
			std::vector<Eigen::SparseMatrix<float> > layers;
			layers.reserve(maxLayers[j]);
			std::vector<float> bias;
			bias.reserve(maxLayers[j]);

			//Read Layer
			auto startLayer = std::chrono::high_resolution_clock::now();
			for(int k = 0; k < maxLayers[j]; ++k){
				if(READTSV) layers.push_back(ReadSparse(StrFileRead(LAYERFILE + std::to_string(Nneuron[i]) + "/n" + std::to_string(Nneuron[i]) + "-l" +  std::to_string(k + 1) + ".tsv"), Nneuron[i], numMNIST,  false));
				//else if(READMAT);
				DNNedges += layers[k].nonZeros();
				bias.push_back(neuralNetBias[i]);
			}
			auto stopLayer = std::chrono::high_resolution_clock::now();
			auto readLayerTime = std::chrono::duration_cast<std::chrono::seconds>(stopLayer - startLayer).count();
			auto readLayerRate = DNNedges / readLayerTime;
			std::cout << "DNN neurons/layer: " << Nneuron[i] << ", layers: " << maxLayers[j] << ", edges: " <<  DNNedges << std::endl;
			std::cout << "Read time (sec):" << readLayerTime << ", Read rate (edges/sec):" << readLayerRate << std::endl;

			//Perform and time challenge
			auto startRun = std::chrono::high_resolution_clock::now();
			Eigen::SparseMatrix<float> scores = inferenceReLUvec(layers, bias, featureVectors);
			auto stopRun = std::chrono::high_resolution_clock::now();
			auto challengeRunTime = std::chrono::duration_cast<std::chrono::seconds>(stopRun - startRun).count();
			auto challengeRunRate = NfeatureVectors * (DNNedges / challengeRunTime);
			std::cout << "Run time (sec):" << challengeRunTime << ", Run rate (edges/sec):" <<challengeRunRate<< std::endl;

			//compute categories from scores.
			Eigen::SparseVector<float> categories = Eigen::MatrixXf(scores).rowwise().sum().sparseView();
			categories = categories.unaryExpr([] (float a) {
					if(a > 0) return 1;
					else return 0;
					});
			if (SAVECAT);
			else{
				Eigen::SparseVector<float> categoryDIff = trueCategories - categories;
				categoryDIff = categoryDIff.pruned();
				if(categoryDIff.nonZeros())
					std::cout << "Challenge FAILED";
				else
					std::cout << "Challenge PASSED" ;
				std::cout << std::endl;
			}
		}
	}
}
