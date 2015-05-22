#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <ctime>
#include "host_matrix.h"
#include "rnn.h"
#include "util.h"
#include "dataset.h"
#include "transforms.h"

#define MAX_EPOCH 1000

using namespace std;

typedef host_matrix<float> mat;

float computeErrRate(const vector<size_t>& ans, const vector<size_t>& output);
void computeLabel(vector<size_t>& result,const mat& outputMat);
void calError(mat& errout,const mat& fin,Transforms* act,Transforms* nex,const mat& delta);


RNN::RNN():_learningRate(0.001),_momentum(0), _method(ALL), _classNum(0){}
RNN::RNN(float learningRate, float momentum,float reg, float variance,Init init, const vector<size_t>& v, Method method, int step):_learningRate(learningRate), _momentum(momentum),_reg(reg), _method(method){
	int numOfLayers = v.size();  
	_classNum = v.back();
	switch(init){
	case NORMAL:
		gn.reset(0,variance);
		for(int i = 0; i < numOfLayers-1; i++){
			Transforms* pTransform;
			if( i < numOfLayers-2 )
				pTransform = new Recursive(v.at(i), v.at(i+1), gn, step);
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
RNN::~RNN(){
	while(!_transforms.empty()){
		delete _transforms.back();
		_transforms.pop_back();
	}
}

void RNN::train(Dataset& data, size_t maxEpoch = MAX_EPOCH, float trainRatio = 0.8, float alpha = 0.98){

	//mat trainSet;
	//mat validSet; 
	//vector<size_t> validLabel;
	//validData.getRecogData(100*batchSize, validSet, validLabel);  

	size_t EinRise = 0;
	float Ein = 1;
	float pastEin = Ein;
	float minEin = Ein;
	float Eout = 1;
	float pastEout = Eout;
	float minEout = Eout;
	
	
	size_t oneEpoch = data.getTrainSentNum();
	//size_t oneEpoch = 20000;
	size_t epochCnt = 0;
	size_t num = 0;
	vector<mat> fin;
	//vector<size_t> validResult;
	for(; epochCnt < maxEpoch; ){   // increment by sentence
		Sentence crtSent = data.getTrainSent();
		fin.clear();
		// push back first word
		num++;
		for (int wordCnt = 0; wordCnt < crtSent.getSize()-1; wordCnt++){
			mat inputMat = crtSent.getWord(wordCnt)->getMatFeature(); // the w2v feature of new input word
			//cout<<"inputmat:"<<inputMat.getRows()<<" "<<inputMat.getCols()<<endl;
			feedForward(inputMat, fin);
			//fout.push_back(tmpOutput);
			backPropagate(_learningRate, _momentum,_reg,fin,crtSent.getWord(wordCnt+1)->getOneOfNOutput(_classNum)); 
			//if(num%1000==0)
			//	cout<<"Iter: "<<num<<endl; 
		}
		for (int i = 0; i < _transforms.size()-1; i++){
			ACT test;
			test=_transforms.at(i)->getAct();
			if(test==RECURSIVE){
				Recursive* temp=(Recursive*)_transforms.at(i);
				temp->resetCounter();
			}
		}
		if(num%5000==0){
			cout<<"Iter:"<<num<<endl;
		}
		/*
		if( num % 2000 == 0 ){
			if(_learningRate==1.0e-4){}
			else if(_learningRate<1.0e-4){_learningRate=1.0e-4;}
			else{_learningRate *= alpha;}
		}
		*/
		//if( num % oneEpoch == 0 ){
		if( num % 20000 == 0 ){
			epochCnt++;
			cout << "epochNum is : "<<epochCnt<<", start validation\n";
			//validResult.clear();
			// calculate validation entropy
			float newAcc = 0;
			for ( int j = 0; j < 10000; j++){
				Sentence validSent = data.getValidSent();
				for (int k = 0; k < validSent.getSize()-1; k++){
					mat validInput = validSent.getWord(k)->getMatFeature();
					feedForward(validInput, fin);
					int tmpAns = validSent.getWord(k+1)->getClassLabel();
					if (tmpAns >= _classNum)
						tmpAns = _classNum -1;
					MatrixXf* tmp = fin.back().getData();
					//if (j == 5000)
					//	cout << *tmp << endl;
					MatrixXf::Index maxR, maxC;
					float maxVal = tmp->maxCoeff(&maxR, &maxC);
					//cout << "maximum : " << maxR <<" " <<  maxC << " " << tmpAns << endl;
					if (maxR == tmpAns){
						newAcc += 1.0/validSent.getSize();
						//cout << "OAO\n";
						//cout << newAcc << endl;
					}
					//newEntropy += log((*tmp)(tmpAns,0));
					//cout << newEntropy << endl;	
				}
			}
			save("temp.mdl");
			cout <<"avg Acc:"<< newAcc << endl;
			data.resetValidSentCtr();
			//predict(validResult, validSet);
			data.resetTrainSentCtr();
			//Eout = computeErrRate(validLabel, validResult);
			/*
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
			*/
		}
		
	}
	cout << "Finished training for " << num << " iterations.\n";
}

void RNN::predict(Dataset& testData, const string& outName = "./model/testOutput.csv"){
	
	ofstream ofs(outName);
	if(!ofs.is_open()){
		cerr << "Fail to open file: " << outName << " !\n";
		exit(1);
	}
	ofs << "Id,Answer\n";

	size_t	testNum = testData.getTestSentNum();
	cout << "Questions:" << testNum/5 << endl;

	vector<mat> fin;
	size_t i = 0;
	while( i < testNum ){
		double tempMin = DBL_MAX;
		size_t minIdx = 0;
		for(size_t j = 0; j < 5; j++ ){
			Sentence testSent = testData.getTestSent();
			++i; 
			fin.clear();
			double crossEntropy = 0.0;
			for (int k = 0; k < testSent.getSize()-1; k++){
				int currentAns = testSent.getWord(k)->getClassLabel();
				if( currentAns == -1 )
					continue;
				mat validInput = testSent.getWord(k)->getMatFeature();
				feedForward(validInput, fin);
				int tmpAns = testSent.getWord(k+1)->getClassLabel();
				MatrixXf* tmp = fin.back().getData();

				//cout << tmpAns << " " ;
				//cout << (*tmp)(tmpAns, 0) << endl; 
				
				// Cross Entropy method
				if(tmpAns != -1)
					crossEntropy -= log((double)((*tmp)(tmpAns, 0)));
				//MatrixXf::Index maxR, maxC;
				//float maxVal = tmp->maxCoeff(&maxR, &maxC);
				//if (maxR == tmpAns){
				//	newAcc += 1.0/validSent.getSize();
				//}
			}
			//cout << crossEntropy <<endl;
			if( tempMin > crossEntropy ){
				tempMin = crossEntropy;
				minIdx = j;
			}
		}
		//cout << i/5 <<" min: " << (char)('a' + minIdx) << endl;
		ofs << i/5 << "," << (char)('a' + minIdx) << endl;
	}

	ofs.close();
}

void RNN::setLearningRate(float learningRate){
	_learningRate = learningRate;
}
void RNN::setMomentum(float momentum){
	_momentum = momentum;
}
void RNN::setReg(float reg){_reg=reg;}
size_t RNN::getInputDimension(){
	return _transforms.front()->getInputDim();
}

size_t RNN::getOutputDimension(){
	return _transforms.back()->getOutputDim();
}

size_t RNN::getNumLayers(){
	return _transforms.size()+1;
}

void RNN::save(const string& fn){
	ofstream ofs(fn);
	if (ofs.is_open()){
		for(size_t i = 0; i < _transforms.size(); i++){
			(_transforms.at(i))->write(ofs);
		}
	}
	ofs.close();
}

bool RNN::load(const string& fn){
	ifstream ifs(fn);
	char buf[50000];
	if(!ifs){return false;}
	else{
		while(ifs.getline(buf, sizeof(buf)) != 0 ){
			string tempStr(buf);
			size_t found = tempStr.find_first_of(">");
			size_t typeBegin = tempStr.find_first_of("<") + 1;
			if(found !=std::string::npos ){
				string type = tempStr.substr(typeBegin, found-typeBegin);
				stringstream ss(tempStr.substr(found+1));
				string rows, cols;
				size_t rowNum, colNum;
				ss >> rows >> cols;
				rowNum = stoi(rows);
				colNum = stoi(cols);
				size_t totalEle = rowNum * colNum;
				float* h_data = new float[totalEle];
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
				Transforms* pTransform;
				
				mat weightMat(h_data, rowNum, colNum);
				if(type == "recursive"){
					ifs.getline(buf, sizeof(buf));
					tempStr.assign(buf);
					found = tempStr.find_first_of(">");
					stringstream ss2(tempStr.substr(found+1));
					size_t step;
					ss2 >> rows >> cols >> step;
					rowNum = stoi(rows);
					colNum = stoi(cols);
					totalEle = rowNum * colNum;
					float* h_data_mem = new float[totalEle];
					for(size_t i = 0; i < rowNum; i++){
						if(ifs.getline(buf, sizeof(buf)) == 0){
							cerr << "Wrong file format!\n";
						}
						tempStr.assign(buf);
						stringstream ss3(tempStr);	
						for(size_t j = 0; j < colNum; j++){
							ss3 >> h_data_mem[ j*rowNum + i ];
						}
					}
					mat memoryMat(h_data_mem, rowNum, colNum);
					pTransform = new Recursive(weightMat, memoryMat, step);
					delete [] h_data_mem;
				}
				else if(type == "softmax"){
					pTransform = new Softmax(weightMat);
				}
				else{
					cerr << "Undefined weight format! \" " << type << " \"\n";
					exit(1);
				}
					
				_transforms.push_back(pTransform);
				delete [] h_data;
			}
		}
	}
	ifs.close();
	return true;
}

void RNN::feedForward(const mat& inputMat,vector<mat>& fout){
	//mat tempInputMat = inputMat;
	fout.resize(_transforms.size()+1);//
	fout[0]=inputMat;
	_transforms.at(0)->forward(fout[1],fout[0]);
	for(size_t i = 1; i < _transforms.size(); i++){
		(_transforms.at(i))->forward(fout[i+1],fout[i] );
		//tempInputMat = outputMat;
	}
}

void RNN::getHiddenForward(mat& outputMat, const mat& inputMat){
	vector<mat> fout;
	fout.resize(1);
	_transforms.at(0)->forward(fout[0],inputMat);
	outputMat=fout.front();
}

//The delta of last layer = _sigoutdiff & grad(errorFunc())
void RNN::backPropagate(float learningRate, float momentum,float regularization,const vector<mat>& fin,const mat& answer){
	vector<mat> err;
	err.resize(_transforms.size());
	err.back() = fin.back() - answer;  //cross entropy gradient
	size_t size=_transforms.size();
	_transforms.back()->backPropagate(fin.at(size-1),err[size-1],learningRate,momentum,regularization); //for softmax
	for(int i=0;i<_transforms.size()-1;i++){
		calError(err[size-2-i],fin.at(size-1-i),_transforms[size-1-i],_transforms[size-2-i],err.at(size-1-i));
		_transforms[size-2-i]->backPropagate(fin.at(size-2-i),err.at(size-2-i),learningRate,momentum,regularization);
	}
/*
	mat errorMat;
	for(int i = _transforms.size()-1; i >= 0; i--){
		(_transforms.at(i))->backPropagate(errorMat, tempMat, learningRate, momentum,regularization);
		tempMat = errorMat;
	}*/
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

void calError(mat& errout,const mat& fin,Transforms* act,Transforms* nex,const mat& delta){
	//modified
	ACT type=nex->getAct();
	mat sigdiff=fin & ((float)1.0-fin);
	mat w(act->getWeight());
	switch(type){
		case RECURSIVE:
		case SIGMOID:
			errout=sigdiff & ( ~w * delta);
			break;
		case SOFTMAX:
		default:
			break;
	}
}

