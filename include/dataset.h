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

#define LABEL_NUM 48
#define OUTPUT_NUM 48
using namespace std;
typedef host_matrix<float> mat;

class Word{
	public:
		Word() : _classLabel(0), _index(0) {}
		Word(int clabel, int index, vector<double> feature):
		_classLabel(clabel), _index(index), _feature(feature) {}
		int getClassLabel() {return _classLabel; }
		int getIndex() {return _index; }
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
	private:
		vector<Word*> _sentence;
};

class Dataset{
	public:
	Dataset();
	
	Dataset(const char* featurePath, const char* classPath, const char* sntPath);
	
	//Dataset(Data data, char mode);
	Dataset(const Dataset& data);
	~Dataset();
	
	//mat getData();	
	size_t getSentenceNum() {return _data.size(); }
	size_t getFeatureDim() {return _featureDim; }
	size_t getWordNum() {return _wordNum;}	
	//vector<size_t> getLabel_vec();
	//mat getLabel_mat();
	
	
	//map<string, int> getLabelMap();
	//map<string, string> getTo39PhonemeMap();
	
	//void   getBatch(int batchSize, mat& batch, mat& batchLabel, bool isRandom);
	//bool   getRecogData(int batchSize, mat& batch, vector<size_t>& batchLabel);  
	//void   getTrainSet(int trainSize, mat& trainData, vector<size_t>& trainLabel);
	//void   getValidSet(int validSize, mat& validData, vector<size_t>& validLabel);
	//void   dataSegment( Dataset& trainData, Dataset& validData, float trainProp = 0.8);
	//void   printLabelMap(map<string, int> labelMap);
	//void   printTo39PhonemeMap(map<string, string>);
	//void   prtPointer(float** input, int r, int c);
	//void   loadTo39PhonemeMap(const char*);
	//void   saveCSV(vector<size_t> testResult);
    //bool  isLabeled( ){ return _isLabeled; }	
private:
	// dataset parameters
	size_t _featureDim;
	size_t _wordNum;
	//mat    outputNumtoBin(int* outputVector, int vectorSize);
	// change 0~47 to a 48 dim mat
	//mat    inputFtreToMat(float** input, int r, int c);	
    // void   prtPointer(float** input, int r, int c);	
	
	// original data
	vector<Sentence> _data;
	unordered_map<string, Word*> _wordMap;
	
};

#endif 
