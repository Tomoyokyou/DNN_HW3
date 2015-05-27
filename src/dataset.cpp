#include "dataset.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdlib> // rand()
#include <ctime>
using namespace std;
typedef host_matrix<float> mat;

Dataset::Dataset(){
	_featureDim = 0;
	_wordNum = 0;
	_sentCtr = 0;
	_trainSentCtr = 0;
	_cutClass = 0;
}

Dataset::Dataset(const char* featurePath, const char* classPath, const char* sntPath, int cutClass){
	// initializing
	_sentCtr = 0;
	_cutClass = cutClass;
	_trainSentCtr = 0;
	cout << "cutClass is " << _cutClass << endl;
	_validSentCtr = 0;
	cout << "inputting word2vec file:\n";	
	ifstream fin(featurePath);
	if(!fin) cout<<"Can't open word2vec file!!!\n";
	cout << "inputting class file:\n";
	ifstream classFin(classPath);
	if (!classFin) cout << "Can't open classfile!!!\n";
	
	fin >> _wordNum;
	fin >> _featureDim;
	cout << "wordNum is : " << _wordNum << endl;
	cout << "featureDim is : " << _featureDim << endl;
	//cout << "start getting features:\n";
	for (int i = 0; i < _wordNum; i++){
		int cLabel = 0;
		float tmp = 0;
		string wordName;
		fin >> wordName;
		string tmpStr;
		//classFin >> tmpStr >> cLabel;
		float* feature = new float[_featureDim];
		//cout << _word[i] << endl;
		for (int j = 0; j < _featureDim; j++){
			//cout << j;
			fin >> tmp;
			feature[j] = tmp;
		}
		mat matFeat(feature, _featureDim, 1);
		delete[] feature;
		Word tmpWord (-1, 0, matFeat);
		_wordMap[wordName] = tmpWord;
		//cout << wordName << " " << i << " " << cLabel << endl;
		//cout << wordName << " " << _wordMap[wordName]->getIndex() << " " << _wordMap[wordName]->getClassLabel() << endl;
	}
	string wordName;
	int cLabel;
	while(classFin >> wordName >> cLabel){
		auto it = _wordMap.find(wordName);
		if (it == _wordMap.end())
			cout << "ERROR::word not in the list!\n";
		it->second.setClassLabel(cLabel - _cutClass);
		//cout << it->first << " " << cLabel << endl;
		if (_classCount.size() == cLabel - _cutClass)
			_classCount.push_back(0);
		if (cLabel - _cutClass >= 0){
			_classCount[cLabel - _cutClass] ++;
			it->second.setIndex(_classCount[cLabel - _cutClass]-1);
		}
	}
	//
	unordered_map<string,Word>::iterator iter;
	for(iter=_wordMap.begin();iter!=_wordMap.end();++iter){
		if (iter->second.getClassLabel() >= 0)
			(iter->second).genMat(_classCount.size(),_classCount[(iter->second).getClassLabel()]);
	}
	//

	//debugging
	/*
	cout <<_wordMap.size()<<"yo"<<endl;
	auto it = _wordMap.begin();
	for (int i = 0 ; it != _wordMap.end(); it ++, i++){
		cout << it->first << " " << i << " ";
		cout << _wordMap[it->first].getIndex() << endl;
		cout << it->second.getClassLabel() << endl;
		//for (int x = 0; x < it->second->getFeatureDim(); x++)
		//	it->second->getMatFeature().print();
		//cout << endl;
	}
	*/
	// end debugging
	fin.close();
	classFin.close();
	
	cout << "inputting sentence file:\n";
	ifstream sntFin(sntPath);
	if (!sntFin) cout << "Can't open sentence file!!!\n";
	string tmpStr;
	Sentence tmpSent;
	int bla = 0;
	while(sntFin >> tmpStr){
		//bla ++;
		if ( _wordMap.find(tmpStr) == _wordMap.end()){
			bla++;
			cout << "new word!!\n";
			Word tmpWord;
			_wordMap[tmpStr] = tmpWord;
			//cout << tmpStr << endl;
		}
		
		auto it = _wordMap.find(tmpStr);
		if (it->second.getClassLabel() >= 0)
			tmpSent.getSent().push_back(&it->second);
		/*
		cout << tmpStr << " ";
		cout << it->first << endl;
		cout << it->second.getIndex() << endl;
		cout << it->second.getFeatureDim() << endl;
		cout << it->second.getClassLabel() << endl;
		*/
		if (tmpStr.compare("</s>") == 0){
			//cout << tmpSent.getSent().size() << " yo "; 
			Sentence toBeStored(tmpSent);
			//cout << "qq" << toBeStored.getSent().size() << "qq "; 
			_data.push_back(toBeStored);
			tmpSent.getSent().clear();
			//cout << toBeStored.getSent().size() << " " << endl;
			// debugging
			/*cout << "sentence size is : " << tmpSent.getSize() << endl;
			for (int x = 0; x < tmpSent.getSize(); x++)
				cout << tmpSent.getWord(x)->getIndex() << " ";
			cout << endl;
			*/
		}
	}
	cout <<"words not in w2v: "<< bla << endl;
	cout <<"total words in map: "<< _wordMap.size() << endl;
	cout << "read sentence done!!!\n";
	sntFin.close();
	dataSegment( 0.95);

};

Dataset::Dataset(const Dataset& d){
	_featureDim = d._featureDim;
	_wordNum = d._wordNum;
	_sentCtr = d._sentCtr;
	_data = d._data;
	_wordMap = d._wordMap;
};

Word::Word(int clabel,int index,mat feature,int ccountsize,int classcountidx):_classLabel(clabel),_index(index),_feature(feature){
	classoutput.resize(ccountsize,1,0);
	MatrixXf* tmp=classoutput.getData();
	if(_classLabel >=0)
		(*tmp)(_classLabel,0)=1;
	wordoutput.resize(classcountidx,1,0);
	tmp=wordoutput.getData();
	if(_index>=0)
		(*tmp)(_index,0)=1;
}
void Word::genMat(int ccountsize,int classcountidx){
	classoutput.resize(ccountsize,1,0);
	MatrixXf* tmp=classoutput.getData();
	if(_classLabel>=0)
		(*tmp)(_classLabel,0)=1;
	wordoutput.resize(classcountidx,1,0);
	tmp=wordoutput.getData();
	if(_index>=0)
		(*tmp)(_index,0)=1;
}

mat Word::getClassOutput(Dataset& d) {
	int classNum = d._classCount.size();
	float* tmpPtr = new float[classNum];
	for (int i = 0; i < classNum; i++)
		tmpPtr[i] = 0;
	/*if (_classLabel < wordNum)
		tmpPtr[_classLabel] = 1;
	else
		tmpPtr[wordNum - 1] = 1;
	*/
	if (_classLabel >= 0)
		tmpPtr[_classLabel] = 1;
	mat temp(tmpPtr, classNum, 1);
	delete [] tmpPtr;
	return temp;
}

mat Word::getWordOutput(Dataset& d) {
	int classNum = d._classCount[_classLabel];
	float* tmpPtr = new float[classNum];
	for (int i = 0; i < classNum; i++)
		tmpPtr[i] = 0;
	/*if (_classLabel < wordNum)
		tmpPtr[_classLabel] = 1;
	else
		tmpPtr[wordNum - 1] = 1;
	*/
	if (_index >= 0)
		tmpPtr[_index] = 1;
	mat temp(tmpPtr, classNum, 1);
	delete [] tmpPtr;
	return temp;
}
mat Word::getMatFeature() {
	return _feature;
}
Sentence Dataset::getSentence() {
	Sentence tmp = _data[_sentCtr];
	_sentCtr++;
	if (_sentCtr == _data.size())
		_sentCtr = 0;
	return tmp;
}
Sentence Dataset::getTrainSent() {
	Sentence tmp = _data[_trainLabel[_trainSentCtr]];
	_trainSentCtr++;
	//cout << "current training sentence is "<< _trainLabel[_trainSentCtr]<< endl;
	if (_trainSentCtr == _trainLabel.size())
		_trainSentCtr = 0;
	return tmp;
}

//
void Dataset::getAllTrainSent(vector<Sentence>& out){
	for(size_t t=0;t<_trainLabel.size();++t){
		if(_data[_trainLabel[t]].getSize()>2)
		out.push_back(_data[_trainLabel[t]]);
	}
}
void Dataset::getAllTestSent(vector<Sentence>& out){
	out=_testData;
}

void Dataset::getAllValidSent(vector<Sentence>& out){
	for(size_t t=0;t<_validLabel.size();++t){
		if(_data[_validLabel[t]].getSize()>2)
		out.push_back(_data[_validLabel[t]]);
	}
}

//
Sentence Dataset::getValidSent() {
	Sentence tmp = _data[_validLabel[_validSentCtr]];
	_validSentCtr++;
	if (_validSentCtr == _validLabel.size())
		_validSentCtr = 0;
	return tmp;
}
Sentence Dataset::getTestSent() {
	Sentence tmp = _testData[_testSentCtr];
	_testSentCtr++;
	if (_testSentCtr == _testData.size())
		_testSentCtr = 0;
	return tmp;
}
void Dataset::dataSegment(float trainProp = 0.95){
	int trainSize = trainProp * _data.size();
	vector<int> tmpLabel;
	for (int i = 0; i < _data.size(); i++)
		tmpLabel.push_back(i);
	random_shuffle ( tmpLabel.begin(), tmpLabel.end() );
	vector<int> tmpTrain(tmpLabel.begin(), tmpLabel.begin() + trainSize);
	vector<int> tmpValid(tmpLabel.begin() + trainSize, tmpLabel.end());
	_trainLabel = tmpTrain;
	_validLabel = tmpValid;
	//cout << _trainLabel.size() << endl;
	//cout << _validLabel.size() << endl;
}
void Dataset::parseTestData(const char* testPath){
	cout << "inputting test file:\n";
	ifstream sntFin(testPath);
	if (!sntFin) cout << "Can't open test file!!!\n";
	string tmpStr;
	Sentence tmpSent;
	int bla = 0;
	while(sntFin >> tmpStr){
		if ( _wordMap.find(tmpStr) == _wordMap.end()){
			//cout << "word out of vocabulary!!\n";
			bla ++;
			//Word tmpWord;
			//tmpWord.setClassLabel(-1);
			//_wordMap[tmpStr] = tmpWord;
			//cout << tmpStr << endl;
			tmpStr = "<unk>";
		}
		
		auto it = _wordMap.find(tmpStr);
		//unordered_map<string, Word*> const_iterator it = _wordMap.find(tmpStr);
		if (it->second.getClassLabel() >= 0)
			tmpSent.getSent().push_back(&it->second);
		/*
		cout << tmpStr << " ";
		cout << it->first << endl;
		cout << it->second.getIndex() << endl;
		cout << it->second.getFeatureDim() << endl;
		cout << it->second.getClassLabel() << endl;
		*/
		if (tmpStr.compare("</s>") == 0){
			//cout << tmpSent.getSent().size() << " yo "; 
			Sentence toBeStored(tmpSent);
			//cout << "qq" << toBeStored.getSent().size() << "qq "; 
			_testData.push_back(toBeStored);
			//toBeStored.print();
			tmpSent.getSent().clear();
			//cout << toBeStored.getSent().size() << " " << endl;
			// debugging
			/*cout << "sentence size is : " << tmpSent.getSize() << endl;
			for (int x = 0; x < tmpSent.getSize(); x++)
				cout << tmpSent.getWord(x)->getIndex() << " ";
			cout << endl;
			*/
		}
	}
	cout << bla << " words are out of vocabulary!\n";
	cout << "read test sentence done!!!\n";
	sntFin.close();

}

