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
class Dataset;
class Word{
	public:
		Word() : _classLabel(-1), _index(0) {}
		Word(int clabel, int index, mat feature):
		_classLabel(clabel), _index(index), _feature(feature) {}
		int getClassLabel() {return _classLabel; }
		int getIndex() {return _index; }
		mat getClassOutput(Dataset& d);
		mat getWordOutput(Dataset& d);
		mat getMatFeature();
		void setClassLabel(int i) {_classLabel = i;}
		void setIndex(int i) {_index = i;}
		int getFeatureDim() {return _feature.getRows(); }
	private:
		int _classLabel;
		int _index; // index inside label
		mat _feature;
};

class Sentence: public Word{
	public:
		Sentence() {}
		vector<Word*>& getSent() {return _sentence; }
		int getSize() {return _sentence.size(); }
		Word* getWord(int i) {return _sentence[i];}
		void print() {
			for (int i = 0; i < getSize(); i++)
				cout << _sentence[i]->getClassLabel() << " ";
			cout << endl;
		}
	private:
		vector<Word*> _sentence;
};

class Dataset{
	friend class Word;
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
	void   resetValidSentCtr() {_validSentCtr = 0;}
	void   resetTestSentCtr()  {_testSentCtr = 0;}
	Sentence getSentence();
	Sentence getTrainSent();
	Sentence getValidSent();
	Sentence getTestSent();
	size_t getSentCtr() {return _sentCtr;}
	size_t getTrainSentNum() {return _trainLabel.size();}
	size_t getValidSentNum() {return _validLabel.size();}
	size_t getTestSentNum() {return _testData.size();}
	const vector<int>& getClassCount() {return _classCount;}
	void dataSegment(float trainProp);
	void parseTestData(const char* testPath);
	private:
	size_t _featureDim;
	size_t _wordNum;
	size_t _sentCtr;
	size_t _trainSentCtr;
	size_t _validSentCtr;
	size_t _testSentCtr;
	vector<int> _trainLabel;
	vector<int> _validLabel;

	vector<Sentence> _data;
	vector<Sentence> _testData;
	unordered_map<string, Word> _wordMap;
    vector<int> _classCount;	
};

#endif 
