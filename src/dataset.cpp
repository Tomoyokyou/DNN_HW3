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
		if ( _wordMap.find(tmpStr) == _wordMap.end()){
			Word tmpWord;
			_wordMap[tmpStr] = &tmpWord;
			//cout << tmpStr << endl;
			bla ++;
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
/*
	_featureDim = d._featureDim;
	_wordNum = d._wordNum;
	_sntNum = d._sntNum;
	_word = new string[_wordNum];
	_data = new float*[_wordNum];
	for (int i = 0; i < _wordNum; i++){
		_word[i] = d._word[i];
		_data[i] = new float[_featureDim];
		for (int j = 0; j < _featureDim; j++){
			_data[i][j] = d._data[i][j];
		}
	}
*/
};

Dataset::~Dataset(){
/*
if (_word != NULL)
		delete[] _word;
	if (_data != NULL){
		for(int i = 0; i < _wordNum; i++){
			delete[] _data[i];
		}
		delete[] _data;
	}
	*/
};
/*
void Dataset::saveCSV(vector<size_t> testResult){
	
	string name, phoneme;
	ofstream fout("Prediction.csv");
	if(!fout){
		cout<<"Can't write the file!"<<endl;
	}
	fout<<"Id,Prediction\n";
	cout<<testResult.size()<<endl;
	for(size_t i = 0;i<testResult.size();i++){
		name = *(_testDataNameMatrix+i);
		fout<<name<<",";
		for(map<string,int>::iterator it = _labelMap.begin();it!=_labelMap.end();it++){
			if(it->second==testResult.at(i)){
				phoneme = it->first;
	//			cout<<phoneme<<endl;
				break;
			}
		}
		//	map<string, string>iterator it2 = _To39PhonemeMap.find(phoneme);
			phoneme = _To39PhonemeMap.find(phoneme)->second;

		fout<<phoneme<<endl;
	
	}	
	fout.close();
}

*/

//Get function
/*
mat Dataset::getData(){
	cout << "dimension of data: " << _featureDim << "*" << _dataNum << endl;
	return inputFtreToMat(_data, _featureDim, _dataNum);
}
vector<size_t> Dataset::getLabel_vec(){
	vector<size_t> tmp;
	for (int i = 0; i < _dataNum; i ++){
		tmp.push_back(_label[i]);
	}
	return tmp;
}
mat Dataset::getLabel_mat(){
	return outputNumtoBin(_label, _dataNum);
}
size_t Dataset::getDataNum(){ return _dataNum; }
size_t Dataset::getFeatureDim(){ return _featureDim; }
*/
//map<string, int> Dataset::getLabelMap(){return _labelMap;}
//map<string, string> Dataset::getTo39PhonemeMap(){return _To39PhonemeMap;}

//Load function
/*
void Dataset::loadTo39PhonemeMap(const char* mapFilePath){
	ifstream fin(mapFilePath);
	if(!fin) cout<<"Can't open the file!\n";
	string s, sKey, sVal;//For map
	while(getline(fin, s)){
		 int pos = 0;
		 int initialPos = 0;
		int judge = 1;
		while(pos!=string::npos){
				
			pos = s.find("\t", initialPos);
			if(judge==1) sKey = s.substr(initialPos, pos-initialPos);
			else
			{
				sVal = s.substr(initialPos, pos-initialPos);
		//		cout<<sKey<<" "<<sVal<<endl;
				_To39PhonemeMap.insert(pair<string,string>(sKey,sVal));
			}
			initialPos = pos+1;
//			pos=s.find("\t", initialPos);
			judge++;
		}
	}
	fin.close();
}

//Print function
void Dataset::printTo39PhonemeMap(map<string, string> Map){
	map<string, string>::iterator MapIter;
	for(MapIter = Map.begin();MapIter!=Map.end();MapIter++){
		cout<<MapIter->first<<"\t"<<MapIter->second<<endl;	
	}
}	
void   Dataset::printLabelMap(map<string, int> Map){
	map<string, int>::iterator labelMapIter;
	for(labelMapIter = Map.begin();labelMapIter!=Map.end();labelMapIter++){
		cout<<labelMapIter->first<<" "<<labelMapIter->second<<endl;
	}
	
}
*/
/*
bool Dataset::getRecogData(int batchSize, mat& batch, vector<size_t>& batchLabel){
	// use shuffled trainX to get batch sequentially
	float** batchFtre = new float*[batchSize];
	if (_recogCtr + batchSize > _dataNum ){
		batchSize = _dataNum - _recogCtr;
		cout << "reaches the bottom of data, will reduce batchSize to " << batchSize << endl;
	}	
	batchLabel.clear();
		for (int i = 0; i < batchSize; i++){
			batchFtre[i] = _data[ _recogCtr ];
			batchLabel.push_back( _label[ _recogCtr ] );
			_recogCtr ++;
		}
	batch = inputFtreToMat( batchFtre, _featureDim, batchSize);
	//batchLabel = outputNumtoBin( batchOutput, batchSize );
	// free tmp pointers
	delete[] batchFtre;
	batchFtre = NULL;
	if (_recogCtr == _dataNum ){
		_recogCtr = 0;
		return false;
	}
	return true;
}
void Dataset::getBatch(int batchSize, mat& batch, mat& batchLabel, bool isRandom){
	// use shuffled trainX to get batch sequentially
	float** batchFtre = new float*[batchSize];
	int*    batchOutput = new int[batchSize];
	if (isRandom == false){
		for (int i = 0; i < batchSize; i++){
			batchFtre[i] = _data[ _batchCtr % _dataNum ];
			batchOutput[i] = _label[ _batchCtr % _dataNum];
			_batchCtr ++;
		}
	}
	else{
		// random initialize indices for this batch	
	
		int* randIndex = new int [batchSize];
		for (int i = 0; i < batchSize; i++){
			randIndex[i] = rand() % _dataNum; 
		}
		for (int i = 0; i < batchSize; i++){
			batchFtre[i] = _data[ randIndex[i] ];
			batchOutput[i] = _label[ randIndex[i] ];
		}
		delete[] randIndex;
		randIndex = NULL;
	}
	// convert them into mat format
	batch = inputFtreToMat( batchFtre, _featureDim, batchSize);
	batchLabel = outputNumtoBin( batchOutput, batchSize );
	// free tmp pointers
	delete[] batchOutput;
	delete[] batchFtre;
	batchOutput = NULL;
	batchFtre = NULL;
*/
// for debugging, print both matrices
	/*
	cout << "This is the feature matrix\n";
	batch.print();
	cout << "from trainX pointer:\n";
	prtPointer(batchFtre, _numOfLabel, batchSize);
	cout << "This is the label matrix\n";
	batchLabel.print();
	*/
//}
/*
void Dataset::getTrainSet(int trainSize, mat& trainData, vector<size_t>& trainLabel){
	if (_trainSetFlag == true){
		trainData = trainMat;
		return;
	}
	if (trainSize > _trainSize){
		cout << "requested training set size overflow, will only output "
		     << _trainSize << " training sets.\n";
		trainSize = _trainSize;
	}
	trainLabel.clear();
	// random initialize
		
	int* randIndex = new int [trainSize];
	for (int i = 0; i < trainSize; i++){
		if (trainSize == _trainSize)
			randIndex[i] = i;
		else
			randIndex[i] = rand() % _trainSize; 
	}
	float** trainFtre = new float*[trainSize];
	for (int i = 0; i < trainSize; i++){
		trainFtre[i] = _trainX[ randIndex[i] ];
		trainLabel.push_back( _trainY[ randIndex[i] ] );
	}
	trainData = inputFtreToMat(trainFtre, getInputDim(), trainSize);
	
	_trainSetFlag = true;
	trainMat = trainData;
	//cout << "get Train Set:\n";
	//trainData.print();
	delete[] randIndex;
	delete[] trainFtre;
	randIndex = NULL;
	trainFtre = NULL;
}

void Dataset::getValidSet(int validSize, mat& validData, vector<size_t>& validLabel){
	if (_validSetFlag == true){
		validData = validMat;
		return;
	}
	if (validSize > _validSize){
		cout << "requested valid set size is too big, can only feed in " << _validSize << " data.\n";
	validSize = _validSize;
	}
	validLabel.clear();
	// random choose index
	cout << "validate size is : " << validSize << " " << _validSize << endl;
	int* randIndex = new int [validSize];
	for (int i = 0; i < validSize; i++){
		if (validSize == _validSize)
			randIndex[i] = i;
		else
			randIndex[i] = rand() % _validSize; 
	}
	float** validFtre = new float*[validSize];
	for (int i = 0; i < validSize; i++){
		validFtre[i] = _validX[ randIndex[i] ];
		validLabel.push_back( _validY[ randIndex[i] ] );
	}
	validData = inputFtreToMat(validFtre, getInputDim(), validSize);
	
	_validSetFlag = true;
	validMat = validData;
	delete[] validFtre;
	delete[] randIndex;
	validFtre = NULL;
	randIndex = NULL;
}
*/


/*
void Dataset::dataSegment( Dataset& trainData, Dataset& validData, float trainProp){
	
	cout << "start data segmenting:\n";
	cout << "num of data is "<< _dataNum << endl;
	// segment data into training and validating set
	trainData._dataNum = trainProp * _dataNum;
	validData._dataNum = _dataNum - trainData._dataNum;
	trainData._featureDim = _featureDim;
	validData._featureDim = _featureDim;
	if (_isLabeled == false){
		cerr << "this file is not labeled, data is not segmented\n";
		return;
	}

	trainData._notOrig = true;
	validData._notOrig = true;
	//create random permutation
	vector<int> randIndex;
	
	for (int i = 0; i < _dataNum; i++){
		randIndex.push_back( i );
	}
	random_shuffle(randIndex.begin(), randIndex.end());
	// 
	
	cout << "put feature into training set\n";
	cout << "trainingsize = " << trainData._dataNum <<endl;
	trainData._data = new float*[trainData._dataNum];
	trainData._label = new int[trainData._dataNum];
	for (int i = 0; i < trainData._dataNum; i++){
		trainData._data[i] = _data[ randIndex[i] ]; 
		trainData._label[i] = _label[ randIndex[i] ];  // depends on ahpan
	}
	cout << "put feature into validating set\n";
	cout << "validatingsize = " << validData._dataNum <<endl;
	
	validData._data = new float*[validData._dataNum];
	validData._label = new int[validData._dataNum];
	for (int i = 0; i < validData._dataNum; i++){
		validData._data[i] = _data [ randIndex[ trainData._dataNum + i] ];
		validData._label[i] = _label[ randIndex[ trainData._dataNum + i] ];
	}
*/	
	// debugging, print out train x y valid x y
	/*
	prtPointer(_trainX, _numOfLabel, _trainSize);
	prtPointer(_validX, _numOfLabel, _validSize);
	
	cout << "print train phoneme:\n";
	for (int i = 0; i < _trainSize; i++)
		cout << _trainY[i] << " ";
	cout << "print valid phoneme:\n";
	for (int i = 0; i < _validSize; i++)
		cout << _validY[i] << " ";
	*/
//}
/*
mat Dataset::outputNumtoBin(int* outputVector, int vectorSize)
{
	float* tmpVector = new float[ vectorSize * LABEL_NUM ];
	for (int i = 0; i < vectorSize; i++){
		for (int j = 0; j < LABEL_NUM; j++){
			*(tmpVector + i*LABEL_NUM + j) = (outputVector[i] == j)?1:0;
		}
	}

	mat outputMat(tmpVector, LABEL_NUM, vectorSize);
	delete[] tmpVector;
	tmpVector = NULL;
	return outputMat;
}
mat Dataset::inputFtreToMat(float** input, int r, int c){
	// r shall be the number of Labels
	// c shall be the number of data
	//cout << "Ftre to Mat size is : " << r << " " << c<<endl;
	//cout << "size is : " << r << " " << c<<endl;
	float* inputReshaped = new float[r * c];
	for (int i = 0; i < c; i++){
		for (int j = 0; j < r; j++){
			// *(inputReshaped + i*r + j) = *(*(input + i) +j);
			*(inputReshaped + i*r + j) = input[i][j];
		}
	}
	mat outputMat(inputReshaped, r, c);
	delete[] inputReshaped;
	inputReshaped = NULL;
	return outputMat;
}
*/
/*
void Dataset::prtPointer(float** input, int r, int c){
	//cout << "this prints the pointer of size: " << r << " " << c << endl;
	for (int i = 0; i < c; i++){
		cout << i << endl;
		for(int j = 0; j < r; j++){
			cout <<input[i][j]<<" ";
			if ((j+1)%5 == 0) cout <<endl;
		}
		cout <<endl;
	}
	return;
}
*/

