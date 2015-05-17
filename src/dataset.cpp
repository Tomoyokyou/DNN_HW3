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
}

Dataset::Dataset(const char* featurePath, const char* classPath, const char* sntPath){
	// initializing
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
		double tmp = 0;
		string wordName;
		fin >> wordName;
		string tmpStr;
		classFin >> tmpStr >> cLabel;
		vector<double> feature;
		//cout << _word[i] << endl;
		for (int j = 0; j < _featureDim; j++){
			//cout << j;
			fin >> tmp;
			feature.push_back(tmp);
		}
		Word tmpWord (cLabel, i, feature);
		_wordMap[wordName] = &tmpWord;
		//cout << wordName << " " << i << " " << cLabel << endl;
		//cout << wordName << " " << _wordMap[wordName]->getIndex() << " " << _wordMap[wordName]->getClassLabel() << endl;
	}
	//cout << _wordMap.size()<< endl;
	fin.close();
	classFin.close();
	cout << "inputting sentence file:\n";
	ifstream sntFin(sntPath);
	if (!sntFin) cout << "Can't open sentence file!!!\n";
	string tmpStr;
	Sentence tmpSent;
	int bla = 0;
	while(sntFin >> tmpStr){
		bla ++;
		if ( _wordMap.find(tmpStr) == _wordMap.end()){
			Word tmpWord;
			_wordMap[tmpStr] = &tmpWord;
			//cout << tmpStr << endl;
		}
		tmpSent.getSent().push_back(_wordMap[tmpStr]);
		//cout << tmpStr << " ";
		//cout << _wordMap[tmpStr]->getIndex() << endl;
		//cout << tmpStr << " ";
		if (tmpStr.compare("</s>") == 0){
			//cout << tmpSent.getSent().size() << " yo "; 
			Sentence toBeStored(tmpSent);
			//cout << "qq" << toBeStored.getSent().size() << "qq "; 
			_data.push_back(toBeStored);
			tmpSent.getSent().clear();
			//cout << toBeStored.getSent().size() << " " << endl;
		}
	}
	cout <<"words not in w2v: "<< bla << endl;
	cout <<"total words in map: "<< _wordMap.size() << endl;
	cout << "read sentence done!!!\n";
	sntFin.close();
	

};

Dataset::Dataset(const Dataset& d){
	_featureDim = d._featureDim;
	_wordNum = d._wordNum;
	_sentCtr = d._sentCtr;
	_data = d._data;
	_wordMap = d._wordMap;
};

mat Word::getOneOfNOutput(int wordNum) {
	float* tmpPtr = new float[wordNum];
	for (int i = 0; i < wordNum; i++)
		tmpPtr[i] = 0;
	tmpPtr[_classLabel] = 1;
	mat temp(tmpPtr, wordNum, 1);
	return temp;
}

mat Word::getMatFeature() {
	float* tmpPtr = new float[_feature.size()];
	for (int i = 0; i < _feature.size(); i++)
		tmpPtr[i] = _feature[i];
	mat temp(tmpPtr, _feature.size(), 1);
	return temp;
}
Sentence Dataset::getSentence() {
	Sentence tmp = _data[_sentCtr];
	_sentCtr++;
	if (_sentCtr == _data.size())
		_sentCtr = 0;
	return tmp;
}

