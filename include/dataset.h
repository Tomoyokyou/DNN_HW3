#ifndef DATASET_H_
#define DATASET_H_

#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <iostream>
#include <functional>
#include <unordered_map>
#include "host_matrix.h"

using namespace std;
typedef host_matrix<float> mat;

class Word{
	public:
		Word() : _classLabel(0), _index(0) {}
		Word(int clabel, int index, vector<double> feature):
		_classLabel(clabel), _index(index), _feature(feature) {}
		int getClassLabel() {return _classLabel; }
		int getIndex() {return _index; }
		mat getOneOfNOutput(int wordNum);
		mat getMatFeature();
		vector<double>& getFeature() {return _feature; }
		int getFeatureDim() {return _feature.size(); }
	private:
		int _classLabel;
		int _index; // self index, from 0 ~ _wordNum
		vector<double> _feature;
};

class Sentence: public Word{
	public:
		Sentence() {}
		vector<Word*>& getSent() {return _sentence; }
		int getSize() {return _sentence.size(); }
		Word* getWord(int i) {return _sentence[i];}
	private:
		vector<Word*> _sentence;
};

class Dataset{
	public:
	Dataset();
	
	Dataset(const char* featurePath, const char* classPath, const char* sntPath);
	
	Dataset(const Dataset& data);
	~Dataset() {}
	
	size_t getSentenceNum() {return _data.size(); }
	size_t getFeatureDim() {return _featureDim; }
	size_t getWordNum() {return _wordNum;}	
	void   resetSentCtr() {_sentCtr = 0;}
	void   resetTrainSentCtr() {_trainSentCtr = 0;}
	Sentence getSentence();
	Sentence getTrainSent();
	size_t getSentCtr() {return _sentCtr;}
	size_t getTrainSentNum() {return _trainLabel.size();}
	void dataSegment(float trainProp);
private:
	size_t _featureDim;
	size_t _wordNum;
	size_t _sentCtr;
	size_t _trainSentCtr;
	vector<int> _trainLabel;
	vector<int> _validLabel;

	vector<Sentence> _data;
	unordered_map<string, Word*> _wordMap;
	
};

#endif 
