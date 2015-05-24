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


RNN::RNN():_learningRate(0.001),_momentum(0), _method(ALL){}
RNN::RNN(float learningRate, float momentum,float reg, float variance,Init init, const vector<size_t>& v, Method method, int step, Dataset& data):_learningRate(learningRate), _momentum(momentum),_reg(reg), _method(method){
	int numOfLayers = v.size();  
	vector<int> ClassCount = data.getClassCount();
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
		for (int i = 0; i < ClassCount.size(); i++){
			Transforms* pTransform;
			pTransform = new Softmax(v.at(numOfLayers-2), ClassCount[i], gn);
			_outSoftmax.push_back(pTransform);
		//	cout << v.at(numOfLayers-2) << " " << ClassCount[i] <<endl;
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
	vector< pair<vector<mat>,vector<mat> > > forwardSet;
	vector<int> wordClassLabel;
	vector<mat> ans(2);
	clock_t tf=0;
	clock_t tb=0;
	clock_t t;
	//vector<size_t> validResult;
	for(; epochCnt < maxEpoch; ){   // increment by sentence
		Sentence crtSent = data.getTrainSent();
		fin.clear();
		forwardSet.clear();
		wordClassLabel.clear();
		// push back first word
		num++;
		t=clock();
		for (int wordCnt = 0; wordCnt < crtSent.getSize()-1; wordCnt++){
			// check whether OOV or not
			int nextLabel = crtSent.getWord(wordCnt+1)->getClassLabel();
			int tmpLabel = crtSent.getWord(wordCnt)->getClassLabel();
			if (tmpLabel == -1 || nextLabel == -1) continue;
			wordClassLabel.push_back(nextLabel);
			feedForward(crtSent.getWord(wordCnt)->getMatFeature(), fin, nextLabel);
			// store all forward output 
			ans[0]=crtSent.getWord(wordCnt+1)->getClassOutput(data);
			//cout<<"ans0 "<<ans[0].getRows()<<endl;
			ans[1]=crtSent.getWord(wordCnt+1)->getWordOutput(data);
			//cout<<"ans1 "<<ans[1].getRows()<<endl;
			forwardSet.push_back(pair<vector<mat>,vector<mat>>(fin,ans));
		}
		tf+=clock()-t;
		t=clock();
		//TODO for all sequence
		backPropagate(_learningRate,_reg,forwardSet, wordClassLabel);
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
			cout << "iterNum is : "<<epochCnt<<", start validation\n";
			//validResult.clear();
			// calculate validation entropy
			float newAcc = 0;
			for ( int j = 0; j < 10000; j++){
				Sentence validSent = data.getValidSent();
				for (int k = 0; k < validSent.getSize()-1; k++){
					if (validSent.getWord(k)->getClassLabel() == -1 ||
					    validSent.getWord(k+1)->getClassLabel() == -1) continue;
					mat validInput = validSent.getWord(k)->getMatFeature();
					int tmpLabel = validSent.getWord(k+1)->getClassLabel();
					feedForward(validInput, fin, tmpLabel);
					int tmpClassAns = validSent.getWord(k+1)->getClassLabel();
					int tmpWordAns = validSent.getWord(k+1)->getIndex();
					if (tmpClassAns == -1) continue;
					// word entropy
					MatrixXf* tmp = fin.back().getData();
					MatrixXf::Index w_maxR, w_maxC, c_maxR, c_maxC;
					float maxVal = tmp->maxCoeff(&w_maxR, &w_maxC);
					//cout << "maximum : " << maxR <<" " <<  maxC << " " << tmpAns << endl;
					MatrixXf* ClassTmp = fin.back().getData();
					maxVal = ClassTmp->maxCoeff(&c_maxR, &c_maxC);
					if (w_maxR == tmpWordAns && c_maxR == tmpClassAns ){
						newAcc += 1.0/validSent.getSize();
					}

					//newEntropy += log((*tmp)(tmpAns,0));
					//cout << newEntropy << endl;	
				}
				// reset counter
				for (int i = 0; i < _transforms.size()-1; i++){
					ACT test;
					test=_transforms.at(i)->getAct();
					if(test==RECURSIVE){
						Recursive* temp=(Recursive*)_transforms.at(i);
						temp->resetCounter();
					}
				}
			}
			//save("temp.mdl");
			cout <<"avg Acc:"<< newAcc/10000 << endl;
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
	//vector<mat> fout;
	//feedForward(inputMat,fout); // modified
	//computeLabel(result, fout.back());
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

void RNN::feedForward(const mat& inputMat,vector<mat>& fout, int classLabel){
	//mat tempInputMat = inputMat;
	fout.resize(_transforms.size()+2);//
	fout[0]=inputMat;
	_transforms.at(0)->forward(fout[1],fout[0]);
	for(size_t i = 1; i < _transforms.size(); i++){
		(_transforms.at(i))->forward(fout[i+1],fout[i] );
	}
	_outSoftmax[classLabel]->forward(fout[_transforms.size()+1], fout[_transforms.size()-1]);
}

void RNN::getHiddenForward(mat& outputMat, const mat& inputMat){
	vector<mat> fout;
	fout.resize(1);
	_transforms.at(0)->forward(fout[0],inputMat);
	outputMat=fout.front();
}

void RNN::backPropagate(float learningRate,float regularization,const vector<pair<vector<mat>,vector<mat>> >& fromForward,const vector<int>& classLabel){
	//NOTE fromForward first= forward+class+word    second=0 class 1 word
	size_t fsize=fromForward.size();
	size_t tsize=_transforms.size();
	ACT testType;
	vector<mat> delta(_transforms.size());
	mat delta1,delta2;
	mat classdiff;
	assert(fsize==classLabel.size());
	for(size_t t=0;t<fsize;++t){
		classdiff=(fromForward[fsize-1-t].first.at(tsize-1))&((float)1.0-fromForward[fsize-1-t].first.at(tsize-1));
		delta1=(fromForward[fsize-1-t].first).at(tsize)-fromForward[fsize-1-t].second.at(0);
		delta2=(fromForward[fsize-1-t].first).at(tsize+1)-fromForward[fsize-1-t].second.at(1);
		_transforms.back()->backPropagate(fromForward[fsize-1-t].first.at(tsize-1),delta1,learningRate,regularization);
		_outSoftmax[classLabel[fsize-1-t]]->backPropagate(fromForward[fsize-1-t].first.at(tsize-1),delta2,learningRate,regularization);
		delta1=classdiff & (~(_transforms.back()->getWeight())*delta1);
		delta2=classdiff & (~(_outSoftmax[classLabel[fsize-1-t]]->getWeight())*delta2);
		delta.front()=delta1+delta2;
		//_transforms.back()->backPropagate(fromForward[fsize-1-t].first.at(tsize-1),delta[0],learningRate,regularization);
		for(size_t k=1;k<tsize;++k){
				//calError(delta[k],fromForward[fsize-1-t].first.at(tsize-k),_transforms[tsize-k],_transforms[tsize-1-k],delta[k-1]);
				_transforms[tsize-1-k]->backPropagate(fromForward[fsize-1-t].first.at(tsize-1-k),delta[0],learningRate,regularization);
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

