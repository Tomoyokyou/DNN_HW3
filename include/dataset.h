#ifndef DATASET_H_
#define DATASET_H_

#include <string>
#include <vector>
#include <fstream>
#include <map>
#include "host_matrix.h"

#define LABEL_NUM 48
#define OUTPUT_NUM 48
using namespace std;
typedef host_matrix<float> mat;
/*
typedef struct{
	
	const char* trainPath;
	size_t trainDataNum;
	const char* testPath;
	size_t testDataNum;
	const char* labelPath;
	size_t labelDataNum;
	size_t labelNum;
	size_t inputDim;
	size_t outputDim;
	size_t phonemeNum;
	int frameRange;
}Data;
*/
class Dataset{
	public:
	Dataset();
	
	Dataset(const char* dataPath);
	
	//Dataset(Data data, char mode);
	Dataset(const Dataset& data);
	~Dataset();
	
	mat getData();	
	size_t getDataNum();
	size_t getFeatureDim();
	vector<size_t> getLabel_vec();
	mat getLabel_mat();
	
	
	//map<string, int> getLabelMap();
	//map<string, string> getTo39PhonemeMap();
	
	void   getBatch(int batchSize, mat& batch, mat& batchLabel, bool isRandom);
	bool   getRecogData(int batchSize, mat& batch, vector<size_t>& batchLabel);  
	//void   getTrainSet(int trainSize, mat& trainData, vector<size_t>& trainLabel);
	//void   getValidSet(int validSize, mat& validData, vector<size_t>& validLabel);
	void   dataSegment( Dataset& trainData, Dataset& validData, float trainProp = 0.8);
	//void   printLabelMap(map<string, int> labelMap);
	//void   printTo39PhonemeMap(map<string, string>);
	//void   prtPointer(float** input, int r, int c);
	//void   loadTo39PhonemeMap(const char*);
	//void   saveCSV(vector<size_t> testResult);
    bool  isLabeled( ){ return _isLabeled; }	
private:
	// dataset parameters
	size_t _featureDim; //input Dim : 39 or 69
	size_t _dataNum;
	size_t _labelNum;
	int _frameRange; //Used for frame ??
	long int _batchCtr;	
	int _recogCtr;
	bool _notOrig;
	bool _isLabeled;
	mat    outputNumtoBin(int* outputVector, int vectorSize);
	// change 0~47 to a 48 dim mat
	mat    inputFtreToMat(float** input, int r, int c);	
    // void   prtPointer(float** input, int r, int c);	
	
	// original data
	string* _name;
	float** _data;
	int* _label; // output phoneme changed to integer
	
	map<string, int> _labelMap; //Map phoneme to int
	map<string, string> _To39PhonemeMap; //Map the output to 39 dimension
	
};

#endif 
