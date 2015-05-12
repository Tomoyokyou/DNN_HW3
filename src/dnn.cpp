#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <ctime>
#include "host_matrix.h"
#include "dnn.h"
#include "util.h"
#include "dataset.h"

#define MAX_EPOCH 1000

using namespace std;

typedef host_matrix<float> mat;

float computeErrRate(const vector<size_t>& ans, const vector<size_t>& output);
void computeLabel(vector<size_t>& result,const mat& outputMat);

DNN::DNN():_learningRate(0.001),_momentum(0), _method(ALL){}
DNN::DNN(float learningRate, float momentum, float variance,Init init, const vector<size_t>& v, Method method):_learningRate(learningRate), _momentum(momentum), _method(method){
	int numOfLayers = v.size();
	switch(init){
	case NORMAL:
		gn.reset(0,variance);
		for(int i = 0; i < numOfLayers-1; i++){
			Transforms* pTransform;
			if( i < numOfLayers-2 )
				pTransform = new Sigmoid(v.at(i), v.at(i+1), gn);
			else
				pTransform = new Softmax(v.at(i), v.at(i+1), gn);
			_transforms.push_back(pTransform);
		}
		break;
	case UNIFORM:
	case RBM:
	default:
		for(int i = 0; i < numOfLayers-1; i++){
			Transforms* pTransform;
			if( i < numOfLayers-2 )
				pTransform = new Sigmoid(v.at(i), v.at(i+1), variance);
			else
				pTransform = new Softmax(v.at(i), v.at(i+1), variance);
			_transforms.push_back(pTransform);
		}
		break;
	}
}
DNN::~DNN(){
	while(!_transforms.empty()){
		delete _transforms.back();
		_transforms.pop_back();
	}
}

void DNN::train(Dataset& labeledData, size_t batchSize, size_t maxEpoch = MAX_EPOCH, float trainRatio = 0.8, float alpha = 0.98){
	if(labeledData.isLabeled() == false){
		cerr << "It is impossible to train unLabeled data.\n";
		return;
	}
	Dataset trainData;
	Dataset validData;

	labeledData.dataSegment(trainData, validData, trainRatio);

	//mat trainSet;
	mat validSet; 
	vector<size_t> validLabel;
	validData.getRecogData(100*batchSize, validSet, validLabel);  

	size_t EinRise = 0;
	float Ein = 1;
	float pastEin = Ein;
	float minEin = Ein;
	float Eout = 1;
	float pastEout = Eout;
	float minEout = Eout;
	
	
	size_t oneEpoch = trainData.getDataNum()/batchSize;
	size_t epochCnt = 0;
	size_t num = 0;
	for(; epochCnt < maxEpoch; num++){
		mat batchData;
		mat batchLabel;

		trainData.getBatch(batchSize, batchData, batchLabel, true);
		mat batchOutput;
		
		feedForward(batchOutput, batchData, true);

		mat lastDelta(batchOutput - batchLabel);
		backPropagate(lastDelta, _learningRate, _momentum); //momentum
		
		if( num % 2000 == 0 ){
			if(_learningRate==1.0e-4){}
			else if(_learningRate<1.0e-4){_learningRate=1.0e-4;}
			else{_learningRate *= alpha;}
		}

		if( num % oneEpoch == 1 ){
			epochCnt++;
			vector<size_t> validResult;
			predict(validResult, validSet);

			Eout = computeErrRate(validLabel, validResult);

			pastEout = Eout;
			if(minEout > Eout){
				minEout = Eout;
				cout << "bestMdl: Error at: " << minEout << endl;  
				if(minEout < 0.5){
					ofstream ofs("best.mdl");
					if (ofs.is_open()){
						for(size_t i = 0; i < _transforms.size(); i++){
							(_transforms.at(i))->write(ofs);
						}
					}
					ofs.close();
				}
			}

			cout.precision(4);
			cout << "Validating error: " << Eout*100 << " %,  Epoch:" << epochCnt <<"\n";
		}
	}
	cout << "Finished training for " << num << " iterations.\n";
	cout << "bestMdl: Error at: " << minEout << endl;  
}

void DNN::predict(vector<size_t>& result, const mat& inputMat){
	mat outputMat(1, 1);
	feedForward(outputMat, inputMat, false);
	computeLabel(result, outputMat);
}

void DNN::setLearningRate(float learningRate){
	_learningRate = learningRate;
}
void DNN::setMomentum(float momentum){
	_momentum = momentum;
}

size_t DNN::getInputDimension(){
	return _transforms.front()->getInputDim();
}

size_t DNN::getOutputDimension(){
	return _transforms.back()->getOutputDim();
}

size_t DNN::getNumLayers(){
	return _transforms.size()+1;
}

void DNN::save(const string& fn){
	ofstream ofs(fn);
	if (ofs.is_open()){
		for(size_t i = 0; i < _transforms.size(); i++){
			(_transforms.at(i))->write(ofs);
		}
	}
	ofs.close();
}

bool DNN::load(const string& fn){
	ifstream ifs(fn);
	char buf[50000];
	if(!ifs){return false;}
	else{
		while(ifs.getline(buf, sizeof(buf)) != 0 ){
			string tempStr(buf);
			size_t found = tempStr.find_first_of(">");
			if(found !=std::string::npos ){
				size_t typeBegin = tempStr.find_first_of("<") + 1;
				string type = tempStr.substr(typeBegin, 7);
				stringstream ss(tempStr.substr(found+1));
				string rows, cols;
				size_t rowNum, colNum;
				ss >> rows >> cols;
				rowNum = stoi(rows);
				colNum = stoi(cols);
				size_t totalEle = rowNum * colNum;
				float* h_data = new float[totalEle];
				float* h_data_bias = new float[rowNum];
				for(size_t i = 0; i < rowNum; i++){
					if(ifs.getline(buf, sizeof(buf)) == 0){
						cerr << "Wrong file format!\n";
					}
					tempStr.assign(buf);
					stringstream ss1(tempStr);	
					for(size_t j = 0; j < colNum; j++){
						ss1 >> h_data[ j*rowNum + i ];
					}
				}
				ifs.getline(buf, sizeof(buf));
				ifs.getline(buf, sizeof(buf));
				tempStr.assign(buf);
				stringstream ss2(tempStr);
				float temp;
				for(size_t i = 0; i < rowNum; i++){
					ss2 >> h_data_bias[i];
				}
				mat weightMat(h_data,rowNum, colNum);
				mat biasMat(h_data_bias,rowNum, 1);		
				
				Transforms* pTransform;
				if(type == "sigmoid")
					pTransform = new Sigmoid(weightMat, biasMat);
				else if(type == "softmax")
					pTransform = new Softmax(weightMat, biasMat);
				else{
					cerr << "Undefined activation function! \" " << type << " \"\n";
					exit(1);
				}
				_transforms.push_back(pTransform);
				delete [] h_data;
				delete [] h_data_bias;
			}
		}
	}
	ifs.close();
	return true;
}

void DNN::feedForward(mat& outputMat, const mat& inputMat, bool train){
	mat tempInputMat = inputMat;
	for(size_t i = 0; i < _transforms.size(); i++){
		(_transforms.at(i))->forward(outputMat, tempInputMat, train);
		tempInputMat = outputMat;
	}
}

void DNN::getHiddenForward(mat& outputMat, const mat& inputMat){
	_transforms.at(0)->forward(outputMat, inputMat, false);
}

//The delta of last layer = _sigoutdiff & grad(errorFunc())
void DNN::backPropagate(const mat& deltaMat, float learningRate, float momentum){
	mat tempMat = deltaMat;
	mat errorMat;
	for(int i = _transforms.size()-1; i >= 0; i--){
		(_transforms.at(i))->backPropagate(errorMat, tempMat, learningRate, momentum);
		tempMat = errorMat;
	}
}


void computeLabel(vector<size_t>& result,const mat& outputMat){
	size_t dim = outputMat.getRows();
	size_t num = outputMat.getCols();
	MatrixXf* optr=outputMat.getData();
	MatrixXf::Index maxidx[num];
	for(size_t t=0;t<num;++t)
		optr->col(t).maxCoeff(&maxidx[t]);
	for(size_t t=0;t<num;++t)
		result.push_back(maxidx[t]);

	/*
	float *data=outputMat.getData();
	for(size_t t=0;t<num;++t){
		max_idx=t*outputMat.getRows();
	for(size_t k=1;k<dim;++k){
		if(data[max_idx]<data[t*dim+k])
			max_idx=t*dim+k;
	}
	result.push_back(max_idx);
	}
	*/
	/*
	thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(outputMat.getData());
	thrust::host_vector<float> h_vec(d_ptr, d_ptr + inputDim*featureNum);
	for(size_t j = 0; j < outputMat.getCols(); j++){
		thrust::host_vector<float>::iterator iter = thrust::max_element(h_vec.begin() + j*inputDim, h_vec.begin() + (j+1)*inputDim);
		unsigned int position = iter - h_vec.begin() - j*inputDim;
		result.push_back(position);
	}
	*/
}

float computeErrRate(const vector<size_t>& ans, const vector<size_t>& output){
	assert(ans.size() == output.size());
	size_t accCount = 0;
	for(size_t i = 0; i < ans.size(); i++){
		if(ans.at(i) == output.at(i)){
			accCount++;
		}
	}
	return 1.0-(float)accCount/(float)ans.size();
}

