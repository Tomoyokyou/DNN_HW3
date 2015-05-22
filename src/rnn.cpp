#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
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

	size_t EinRise = 0;
	float Ein = 1;
	float pastEin = Ein;
	float minEin = Ein;
	float Eout = 1;
	float pastEout = Eout;
	float minEout = Eout;
	
	size_t oneEpoch = data.getTrainSentNum();
	size_t epochCnt = 0;
	size_t num = 0;
	vector<mat> fin;
	vector< pair<vector<mat>,mat > > forwardSet;
	clock_t tf=0;
	clock_t tb=0;
	clock_t t;
	//vector<size_t> validResult;
	for(; epochCnt < maxEpoch; ){   // increment by sentence
		Sentence crtSent = data.getTrainSent();
		fin.clear();
		forwardSet.clear();
		// push back first word
		num++;
		t=clock();
		for (int wordCnt = 0; wordCnt < crtSent.getSize()-1; wordCnt++){
			// check whether OOV or not
			feedForward(crtSent.getWord(wordCnt)->getMatFeature(), fin);
			// store all forward output 
			forwardSet.push_back(pair<vector<mat>,mat>(fin,crtSent.getWord(wordCnt+1)->getOneOfNOutput(_classNum)));
		}
		tf+=clock()-t;
		t=clock();
		//TODO for all sequence
		backPropagate(_learningRate,_reg,forwardSet);
		tb+=clock()-t;
		//reset
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
			cout<<"Feedforward Time: "<<(float)tf/(float)CLOCKS_PER_SEC<<" Backpropagation Time: "<<(float)tb/(float)CLOCKS_PER_SEC<<endl;
			cout<<"Total Time:" <<(float)(tf+tb)/(float)CLOCKS_PER_SEC<<" seconds"<<endl;
			tb=0;tf=0;
		}
		if( num % 20000 == 0 ){
			clock_t test=clock();
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
			cout<<"Time: "<<(float)(clock()-test)/(float)CLOCKS_PER_SEC<<" seconds"<<endl;
			data.resetValidSentCtr();
			//predict(validResult, validSet);
			data.resetTrainSentCtr();
			//Eout = computeErrRate(validLabel, validResult);
		}
		
	}
	cout << "Finished training for " << num << " iterations.\n";
}

void RNN::predict(vector<size_t>& result, const mat& inputMat){
	//mat outputMat(1, 1);
	vector<mat> fout;
	feedForward(inputMat,fout); // modified
	computeLabel(result, fout.back());
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
	}
}

void RNN::getHiddenForward(mat& outputMat, const mat& inputMat){
	vector<mat> fout;
	fout.resize(1);
	_transforms.at(0)->forward(fout[0],inputMat);
	outputMat=fout.front();
}

void RNN::backPropagate(float learningRate,float regularization,const vector<pair<vector<mat>,mat> >& fromForward){
	size_t fsize=fromForward.size();
	size_t tsize=_transforms.size();
	ACT testType;
	vector<mat> delta(_transforms.size());
	for(size_t t=0;t<fsize;++t){
		delta.front()=(fromForward[fsize-1-t].first).back()-fromForward[fsize-1-t].second;
		_transforms.back()->backPropagate(fromForward[fsize-1-t].first.at(tsize-1),delta[0],learningRate,regularization);
		for(size_t k=1;k<tsize;++k){
				calError(delta[k],fromForward[fsize-1-t].first.at(tsize-k),_transforms[tsize-k],_transforms[tsize-1-k],delta[k-1]);
				_transforms[tsize-1-k]->backPropagate(fromForward[fsize-1-t].first.at(tsize-1-k),delta[k],learningRate,regularization);
		}
	}
/*
	vector<mat> err;
	err.resize(_transforms.size());
	err.back() = fin.back() - answer;  //cross entropy gradient
	size_t size=_transforms.size();
	_transforms.back()->backPropagate(fin.at(size-1),err[size-1],learningRate,momentum,regularization); //for softmax
	for(int i=0;i<_transforms.size()-1;i++){
		calError(err[size-2-i],fin.at(size-1-i),_transforms[size-1-i],_transforms[size-2-i],err.at(size-1-i));
		_transforms[size-2-i]->backPropagate(fin.at(size-2-i),err.at(size-2-i),learningRate,momentum,regularization);
	}
*/
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

