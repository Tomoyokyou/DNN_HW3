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
}

Dataset::Dataset(const char* featurePath, const char* classPath, const char* sntPath){
	// initializing
	_sentCtr = 0;
	_trainSentCtr = 0;
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
		classFin >> tmpStr >> cLabel;
		float* feature = new float[_featureDim];
		//cout << _word[i] << endl;
		for (int j = 0; j < _featureDim; j++){
			//cout << j;
			fin >> tmp;
			feature[j] = tmp;
		}
		mat matFeat(feature, _featureDim, 1);
		delete[] feature;
		Word tmpWord (cLabel, i, matFeat);
		_wordMap[wordName] = tmpWord;
		//cout << wordName << " " << i << " " << cLabel << endl;
		//cout << wordName << " " << _wordMap[wordName]->getIndex() << " " << _wordMap[wordName]->getClassLabel() << endl;
	}
	
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
		bla ++;
		if ( _wordMap.find(tmpStr) == _wordMap.end()){
			cout << "new word!!\n";
			Word tmpWord;
			_wordMap[tmpStr] = tmpWord;
			//cout << tmpStr << endl;
		}
		
		auto it = _wordMap.find(tmpStr);
		//unordered_map<string, Word*> const_iterator it = _wordMap.find(tmpStr);
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

mat Word::getOneOfNOutput(int wordNum) {
	float* tmpPtr = new float[wordNum];
	for (int i = 0; i < wordNum; i++)
		tmpPtr[i] = 0;
	/*if (_classLabel < wordNum)
		tmpPtr[_classLabel] = 1;
	else
		tmpPtr[wordNum - 1] = 1;
	*/
	tmpPtr[_index % wordNum ] = 1;
	mat temp(tmpPtr, wordNum, 1);
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
	if (_trainSentCtr == _trainLabel.size())
		_trainSentCtr = 0;
	return tmp;
}
Sentence Dataset::getValidSent() {
	Sentence tmp = _data[_validLabel[_validSentCtr]];
	_validSentCtr++;
	if (_validSentCtr == _validLabel.size())
		_validSentCtr = 0;
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
