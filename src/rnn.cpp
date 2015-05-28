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
#include <cfloat>
#define MAX_EPOCH 1000

using namespace std;

typedef host_matrix<float> mat;

float computeErrRate(const vector<size_t>& ans, const vector<size_t>& output);
void computeLabel(vector<size_t>& result,const mat& outputMat);
void calError(mat& errout,const mat& fin,Transforms* act,Transforms* nex,const mat& delta);
bool readAns(string path,vector<char>& ans);

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
	status();
}
RNN::~RNN(){
	while(!_transforms.empty()){
		delete _transforms.back();
		_transforms.pop_back();
	}
	_outSoftmax.clear();
}
void RNN::status()const
{
	cout<<"*************************************"<<endl;
	cout<<"              RNN status\n";
	cout<<"*************************************"<<endl;
	cout<<"numbers of layers: "<<_transforms.size()<<endl;
	cout<<"numbers of class: "<<_outSoftmax.size()<<endl;
	cout<<"*************************************"<<endl;
	cout<<"Structure:"<<endl;
	for(size_t t=0;t<_transforms.size();++t){
		ACT type=_transforms[t]->getAct();
		if(t!=0)cout<<"->";
		switch(type){
			case SIGMOID: cout<<"Sigmoid("<<_transforms[t]->getOutputDim()<<")";break;
			case RECURSIVE: cout<<"Recursive("<<_transforms[t]->getOutputDim()<<")";break;
			case SOFTMAX: cout<<"Softmax("<<_transforms[t]->getOutputDim()<<")";break;
			default: break;
		}
	}
	cout<<endl;	
	cout<<"*************************************"<<endl;
	cout<<"parameters:\n";
	cout<<" learning rate: "<<_learningRate<<endl;
	cout<<" momentum: "<<_momentum<<endl;
	cout<<" regularization: "<<_reg<<endl;
	cout<<"*************************************"<<endl;
}


void RNN::train(Dataset& data, size_t maxEpoch = MAX_EPOCH, float trainRatio = 0.8, float alpha = 0.98){

	size_t EinRise = 0;
	float Ein = 1;
	float pastEin = Ein;
	float minEin = Ein;
	float Eout = 1;
	float pastEout = Eout;
	float minEout = Eout;
	float maxAcc = 0,temp;
	size_t oneEpoch = data.getTrainSentNum();
	size_t epochCnt = 1;
	size_t num = 0;
	clock_t t=clock();
	vector<mat> fin;
	vector< pair<vector<mat>,vector<mat*> > > forwardSet;
	vector<int> wordClassLabel;
	Sentence crtSent;
	vector<Sentence> trainset;
	vector<Sentence> validset;
	vector<size_t>   validId;
	vector<Sentence> testset;
	data.getAllTrainSent(trainset);
	data.getAllValidSent(validset);
	validId.reserve(validset.size()); 
	for(size_t i = 0; i < validset.size(); i++){
		Sentence tempSent = validset.at(i);
		size_t tempClass = tempSent.getWord(0)->getClassLabel();
		size_t id = 0;
		for (int k = 1; k < tempSent.getSize(); k++){
			if(tempSent.getWord(k)->getClassLabel() <= tempClass){
				tempClass = tempSent.getWord(k)->getClassLabel();
				id = k;
			}
		}
		validId.push_back(id);
	}
	data.getAllTestSent(testset);
	vector<Word*>* wvptr=NULL;

	cout<<"---------------------"<<endl;
	cout<<"-   RNN training  -"<<endl;
	cout<<"---------------------"<<endl;
	cout<<"DATASET:"<<endl;
	cout<<"   training:  "<<trainset.size()<<endl;
	cout<<"   validation:"<<validset.size()<<endl;
	cout<<"   testing:   "<<testset.size()<<endl;
	cout<<"   maxepoch:  "<<maxEpoch<<endl;
	cout<<"---------------------"<<endl;
	for(; epochCnt<maxEpoch+1; epochCnt++ ){   // increment by sentence
		cout<<"EPOCH: "<<epochCnt<<" Highest Accuracy: "<<maxAcc<<endl;
		for(vector<Sentence>::iterator it=trainset.begin();it!=trainset.end();++it){
			wvptr=it->getWordVecPtr();
		//crtSent = data.getTrainSent();
		fin.clear();
		forwardSet.clear();
		wordClassLabel.clear();
		// push back first word
		num++;

		feedForward(*wvptr,forwardSet,wordClassLabel);

		backPropagate(forwardSet, wordClassLabel);
		//reset
		for (int i = 0; i < _transforms.size(); i++){
			_transforms[i]->resetCounter(_learningRate);
		}
		for(int i=0;i<wordClassLabel.size();++i){
			if(!_outSoftmax[wordClassLabel[i]]->isreset()) _outSoftmax[wordClassLabel[i]]->resetCounter(_learningRate);
		}
		if (num % 20000 == 0){
			cout<<"Iter: "<<num<<endl;
			/*Validation*/

			vector<mat> fin;
			size_t numValid = (validset.size() < 2000) ? validset.size() : 2000;
			size_t totalCount = numValid;
			cout << "Num of valid: " << totalCount << endl;

			size_t numAcc = 0;
			for(int j = 0; j < numValid; j++){
				Sentence validSent = validset.at(j);
				if(validSent.getSize() < 2){
					totalCount--;
					continue;
				}
				size_t blank = validId.at(j);
				//cout << "Position of blank: " << blank << endl;

				if(blank < 2){
					totalCount--;
					continue;
				}

				fin.clear();
				
				for (int k = 0; k < blank-1; k++){
					int nextClass = validSent.getWord(k+1)->getClassLabel();
					feedForwardOut(validSent.getWord(k)->getMatPtr(),fin,nextClass);
					if(k == blank-2 && blank >= 2){
						MatrixXf* classtmp = fin[fin.size()-2].getData();
						MatrixXf::Index maxRow, maxCol;
						float max = classtmp->maxCoeff(&maxRow, &maxCol);
						if(maxRow == nextClass)
							numAcc++;
					}
				}
				for (int i = 0; i < _transforms.size(); i++){
					_transforms[i]->resetCounter(_learningRate);
				}
				for(int i=0;i<wordClassLabel.size();++i){
					if(!_outSoftmax[wordClassLabel[i]]->isreset()) _outSoftmax[wordClassLabel[i]]->resetCounter(_learningRate);
				}
			}

			temp = (float)numAcc/totalCount;
			cout << "Validate Acc: " << temp << endl;
			
			if(maxAcc<temp){
				maxAcc=temp;
				if(maxAcc>0.03){
					string modelpath="./model/acc_";
					stringstream s;
					s<<modelpath<<((int)(maxAcc*10000)/(int)10)<<".mdl";
					save(s.str());
					}
			}
			_learningRate=(_learningRate<1e-4)?1e-4:_learningRate*alpha;
			cout<<"current time: "<<(float)(clock()-t)/(float)CLOCKS_PER_SEC<<endl;
		}
		} // iterator
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
	//vector<Sentence> testset;
	//testData.getAllTestSent(testset);
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
				int currentClass = testSent.getWord(k)->getClassLabel();
				int nextClass = testSent.getWord(k+1)->getClassLabel();

				//mat testInput = testSent.getWord(k)->getMatFeature();
				//feedForward(testInput, fin, nextClass);
				feedForwardOut(testSent.getWord(k)->getMatPtr(), fin, nextClass);

				MatrixXf* wordtmp = fin.back().getData();
				MatrixXf* classtmp = fin[fin.size()-2].getData();

				int nextIndex = testSent.getWord(k+1)->getIndex();
				// Cross Entropy method
				if(nextIndex != -1)
					crossEntropy -= log((double)((*wordtmp)(nextIndex, 0)));
				if(nextClass != -1)
					crossEntropy -= log((double)((*classtmp)(nextClass, 0)));
			}
			if( tempMin > crossEntropy ){
				tempMin = crossEntropy;
				minIdx = j;
			}
			for (int i = 0; i < _transforms.size(); i++){
				_transforms[i]->resetCounter(0);
			}
			for(int i=0;i<_outSoftmax.size();++i){
				if(!_outSoftmax[i]->isreset()) _outSoftmax[i]->resetCounter(0);
			}
		}
		//cout << i/5 <<" min: " << (char)('a' + minIdx) << endl;
		ofs << i/5 << "," << (char)('a' + minIdx) << endl;
	}
	testData.resetTestSentCtr();
	ofs.close();
}

void RNN::readPredict(Dataset& testData, vector<char>& pred){
	
	size_t	testNum = testData.getTestSentNum();
	cout << "Questions:" << testNum/5 << endl;
	pred.clear();
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
				int currentClass = testSent.getWord(k)->getClassLabel();
				int nextClass = testSent.getWord(k+1)->getClassLabel();

				//mat testInput = testSent.getWord(k)->getMatFeature();
				//feedForward(testInput, fin, nextClass);
				feedForwardOut(testSent.getWord(k)->getMatPtr(), fin, nextClass);

				MatrixXf* wordtmp = fin.back().getData();
				MatrixXf* classtmp = fin[fin.size()-2].getData();

				int nextIndex = testSent.getWord(k+1)->getIndex();
				// Cross Entropy method
				if(nextIndex != -1)
					crossEntropy -= log((double)((*wordtmp)(nextIndex, 0)));
				if(nextClass != -1)
					crossEntropy -= log((double)((*classtmp)(nextClass, 0)));
			}
			if( tempMin > crossEntropy ){
				tempMin = crossEntropy;
				minIdx = j;
			}
			for (int i = 0; i < _transforms.size(); i++){
				_transforms[i]->resetCounter(0);
			}
			for(int i=0;i<_outSoftmax.size();++i){
				if(!_outSoftmax[i]->isreset()) _outSoftmax[i]->resetCounter(0);
			}
		}
		//cout << i/5 <<" min: " << (char)('a' + minIdx) << endl;
		pred.push_back((char)('a'+minIdx));
	}
	testData.resetTestSentCtr();
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
		ofs << "<outsoftmax>" << endl;
		for(size_t i = 0; i < _outSoftmax.size(); i++){
			(_outSoftmax.at(i))->write(ofs);
		}
	}
	ofs.close();
}

bool RNN::load(const string& fn){
	ifstream ifs(fn);
	char buf[50000];
	if(!ifs){return false;}
	else{
		bool isOutSoftmax = false;
		while(ifs.getline(buf, sizeof(buf))){
			string tempStr(buf);
			size_t found = tempStr.find_first_of(">");
			size_t typeBegin = tempStr.find_first_of("<") + 1;
			if(found !=std::string::npos ){
				string type = tempStr.substr(typeBegin, found-typeBegin);
				if(type == "outsoftmax"){
					isOutSoftmax = true;
					continue;
				}	
				stringstream ss(tempStr.substr(found+1));
				string rows, cols;
				size_t rowNum, colNum;
				ss >> rows >> cols;
				rowNum = stoi(rows);
				colNum = stoi(cols);
				size_t totalEle = rowNum * colNum;
				float* h_data = new float[totalEle];
				for(size_t i = 0; i < rowNum; i++){
					if(!ifs.getline(buf, sizeof(buf))){
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
						if(!ifs.getline(buf, sizeof(buf))){
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
				
				if(isOutSoftmax == false)
					_transforms.push_back(pTransform);
				else
					_outSoftmax.push_back(pTransform);
				delete [] h_data;
			}
		}
	}
	ifs.close();
	return true;
}

//void RNN::feedForward(const mat& inputMat,vector<mat>& fout, int classLabel){
//void RNN::feedForward(mat* inputMat,vector<mat>& fout, int classLabel){
void RNN::feedForward(const vector<Word*>& words,vector<pair<vector<mat>,vector<mat*>>>& out,vector<int>& classout){
	vector<Word*>::const_iterator it;
	vector<mat> fout(_transforms.size()+2);
	vector<mat*> ans(2);
	Word* wp1,*wp2;
	for(size_t t=0;t<words.size()-1;++t){
		wp1=words[t];
		wp2=words[t+1];
		classout.push_back(wp2->getClassLabel());
		fout[0]=*(wp1->getMatPtr());
			_transforms[0]->forward(fout[1],fout[0]);
		for(size_t k=1;k<_transforms.size();++k){
			_transforms[k]->forward(fout[k+1],fout[k]);
		}
		_outSoftmax[classout[t]]->forward(fout[_transforms.size()+1],fout[_transforms.size()-1]);
		//ans[0]=wp2->getClassOutput(data);
		//ans[1]=wp2->getWordOutput(data);
		ans[0]=wp2->getClassOutputPtr();
		ans[1]=wp2->getWordOutputPtr();
		out.push_back(pair<vector<mat>,vector<mat*>>(fout,ans));
	}

}

void RNN::feedForwardOut(mat* inputMat,vector<mat>& fout,int classLabel){
	//mat tempInputMat = inputMat;
	fout.resize(_transforms.size()+2);//
	//fout[0]=inputMat;
	fout[0]=*inputMat;
	//_transforms.at(0)->forward(fout[1],fout[0]);
	_transforms[0]->forward(fout[1],fout[0]);
	for(size_t i = 1; i < _transforms.size(); i++){
		_transforms.at(i)->forward(fout[i+1],fout[i]);
	}
	_outSoftmax[classLabel]->forward(fout[_transforms.size()+1], fout[_transforms.size()-1]);
}

void RNN::getHiddenForward(mat& outputMat, const mat& inputMat){
	vector<mat> fout;
	fout.resize(1);
	_transforms.at(0)->forward(fout[0],inputMat);
	outputMat=fout.front();
}

void RNN::backPropagate(const vector<pair<vector<mat>,vector<mat*>> >& fromForward,const vector<int>& classLabel){
	//NOTE fromForward first= forward+class+word    second=0 class 1 word
	size_t fsize=fromForward.size();
	size_t tsize=_transforms.size();
	ACT testType;
	mat delta1,delta2,classdiff,holdOutput;
	mat softSum(_transforms.back()->getOutputDim(),1,0);
	mat dummy(1,1);
	vector<mat>::const_iterator it;
	vector<mat> error(tsize);
	//vector<mat>::iterator ite=error.begin();
	assert(fsize==classLabel.size());
	for(size_t t=0;t<fsize;++t){
		it = (fromForward[fsize-1-t].first).end();it--;
		delta2=(*it)-*(fromForward[fsize-1-t].second.at(1));it--;
		delta1=(*it)-*(fromForward[fsize-1-t].second.at(0));it--;
		//holdOutput=fromForward[fsize-1-t].first.at(tsize-1);
		//this should update after all been calculated
		((Softmax*)_transforms.back())->accGra(*it,delta1,_learningRate,_reg,_momentum);
		((Softmax*)_outSoftmax[classLabel[fsize-1-t]])->accGra(*it,delta2,_learningRate,_reg,_momentum);
		classdiff=(*it) & ((float)1.0-(*it));it--;
		delta1=classdiff & (_transforms.back()->multWeightInv(delta1));
		delta2=classdiff & (_outSoftmax[classLabel[fsize-1-t]]->multWeightInv(delta2));
		error[0]=delta1+delta2;
		//don't care for recursive;
		//calError(error.back(),*it,_transforms[tsize-2],_transforms[tsize-3],sumD);
		//_transforms[tsize-2]->backPropagate(mat(1,1),error.back(),_learningRate,_reg,_momentum);
		for(size_t t=0;t<_transforms.size()-1;t++){
			_transforms[tsize-2-t]->backPropagate(dummy,error[t],_learningRate,_reg,_momentum);
			if(tsize!=t+2)	calError(error[t+1],*it,_transforms[tsize-2-t],_transforms[tsize-3-t],error[t]);
			else break;
			it--;
			//_transforms[tsize-2-t]->backPropagate(dummy,error[tsize-2-t],_learningRate,_reg,_momentum);
		}
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
	//mat w(act->getWeight());
	switch(type){
		case RECURSIVE:
		case SIGMOID:
			errout=sigdiff & ( act->multWeightInv(delta));
			break;
		case SOFTMAX:
		default:
			break;
	}
}

float RNN::calAcc(string prePath,string ansPath){
	ifstream pre(prePath.c_str());
	ifstream ans(ansPath.c_str());
	if (!pre) cout <<"can't open pre file\n";
	if (!ans) cout << "can't open ans file\n";
	string a, p;
	float acc = 0;
	for (int i = 0; i <= 1040; i++){
		pre >> p;
		ans >> a;
		//cout <<p << " " << a << endl;
		if (p[p.find(",")+1] == a[a.find(",")+1])
			acc++;
	}
	cout << "ground truth acc is " << acc/1040 << endl;
	pre.close();
	ans.close();
	return acc/(float)1040;
}

bool readAns(string path,vector<char>& ans){
	ifstream file(path.c_str());
	if(!file){return false;}
	string s;
	getline(file,s);
	for(;getline(file,s);){
		ans.push_back(s[s.find(",")+1]);
	}
	file.close();
	return true;
}
