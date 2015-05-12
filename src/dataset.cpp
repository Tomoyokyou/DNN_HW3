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
	_featureDim = 39;
	_dataNum = 0;
	_labelNum = 0;
	_frameRange = 0;
	_batchCtr = 0;
	_recogCtr = 0;
	_name = NULL;
	_data = NULL;
	_label = NULL;
	_notOrig = false;
	_isLabeled = false;
}

Dataset::Dataset(const char* dataPath){
	_batchCtr = 0;
	_recogCtr = 0;
	_notOrig = false;
	_frameRange = 0;
	int maxDataNum = 1200000;
	// initializing
	size_t count  = 0, dataCount = 0;
	short split = 0;
	int frameNum = 0;
	string utterName;
	string* tmpName = new string[maxDataNum];	
	float** tmpData = new float*[maxDataNum];
	int*    tmpLabel= new int   [maxDataNum];
	
	bool trainFlag;
	cout << "inputting train feature file:\n";	
	ifstream fin(dataPath);
	if(!fin) cout<<"Can't open train file!!!\n";
	
	string s, tempStr;
	vector<float> tmpVec;
	getline(fin, s); //[training]
	if (s.find("training") != -1){
		cout << "this is a training file\n";
		trainFlag = true;
		_isLabeled = true;
	}
	else _isLabeled = false;
	
	while( getline(fin, s) ){
		if (s.find(">") != -1){   // get frame name
			utterName = s.substr(1, s.find(">")-1);
			frameNum = atoi(s.substr( s.find(">")+1 ).c_str());
		}
		else { cout << "something is wrong" << endl; break; }
		for (int dataCount = 0; dataCount < frameNum; dataCount ++)
		{
		    
			getline(fin, s);
			count++; // datanum counter
			
			// write name
			tmpName[count - 1] = utterName + "_";
			char tmpNum[4];
			int tmp = sprintf(tmpNum, "%d", dataCount+1);
			tmpName[count - 1] += tmpNum;

			if (count != 1)
			{
				tmpData[count - 1] = new float[_featureDim];
			}
			unsigned int pos  = s.find(" ");
			unsigned int initialPos = 1;
			if (trainFlag == true){
				tmpLabel[count - 1] = atoi(s.substr(initialPos, pos-initialPos).c_str());
			}
			s = s.substr(pos + 1); // the label 
			pos = s.find(" ");
			initialPos = 0;
			
			

			split=0;
			while( 1 ){
				split++;
					if (pos == -1)
						tempStr= s.substr(initialPos, s.find("]")-initialPos);
					else
						tempStr= s.substr(initialPos, pos-initialPos);
						
					if (count == 1)
						tmpVec.push_back( atof(tempStr.c_str()) );
					else
						tmpData[count-1][split-1] = atof(tempStr.c_str());
					
					if (pos == -1) break;
				
				initialPos = pos+1;
				pos=s.find(" ", initialPos);
			}
			if (count == 1){
				_featureDim = tmpVec.size();
				tmpData[0] = new float[_featureDim];
				for (int i = 0; i < _featureDim; i++)
					tmpData[0][i] = tmpVec[i];
			}
		}
	}

			
	cout<< "data num is: "<<count<<endl;
	cout << "feature num is: " << _featureDim << endl;
	_dataNum = count;
	// put into true matrix and free pointer
	_name = new string[count];
	_data = new float*[count];
	_label= new int   [count];

	for (int i = 0; i < count; i++){
		_name[i] = tmpName[i];
		_label[i] = tmpLabel[i];
		_data[i] = tmpData[i];
	}
	delete[] tmpName;
	delete[] tmpLabel;
	delete[] tmpData;
};

/*
Dataset::Dataset(Data data, char mode){
	
	switch(mode){
	case 'F':
	case 'f':
		{	
		//Set private members
		_numOfTrainData = data.trainDataNum;
		_numOfTestData = data.testDataNum;
		_numOfLabel = data.labelNum;
		_numOfPhoneme = data.phonemeNum;
		_featureDimension = data.inputDim;
		_stateDimension = data.outputDim;
		_frameRange = data.frameRange;
			
		size_t  dataCount = 0;
		size_t count = 0;
		size_t split = 0;
		string* tempTrainDataNameMatrix = new string[data.trainDataNum];
		float** tempTrainDataMatrix = new float*[data.trainDataNum];		
		for(int i=0;i<data.trainDataNum;i++){
			tempTrainDataMatrix[i] = new float [data.inputDim];
		}
		
		ifstream fin(data.trainPath);
		if(!fin) cout<<"Can't open the train data!\n";
		else cout<<"Inputting train data!\n";
		string s, tempStr;
		while(getline(fin,s)&&count<data.trainDataNum){
			count++;
			size_t pos = s.find(" ");
			size_t initialPos=0;
			split=0;
			string tmpName;
			while(split<data.inputDim+1){
				dataCount++;
				split++;
				
				tempStr= s.substr(initialPos, pos-initialPos);
				if (split==1){
					*(tempTrainDataNameMatrix+count-1) = tempStr;
				}

				else{
					tempTrainDataMatrix[count-1][split-2] = atof(tempStr.c_str());
				}		
				initialPos = pos+1;
				pos=s.find(" ", initialPos);
			}		
			
		}	
		
		
		//fin.close();
	
		_trainDataNameMatrix = new string[data.trainDataNum];
		_trainDataMatrix = new float*[data.trainDataNum];
		
		for(int i=0;i<data.trainDataNum;i++){
			_trainDataMatrix[i]=new float[data.inputDim*(2*_frameRange+1)];
//			cout<<data.inputDim*(2*_frameRange+1)<<endl;
		}
		for(int i=0;i<data.trainDataNum;i++){
			unsigned int pos = (*(tempTrainDataNameMatrix+i)).find_last_of("_");				
			//cout<<*(tempTrainDataNameMatrix+i)<<endl;
			_trainDataNameMatrix[i]=tempTrainDataNameMatrix[i];
			string str = _trainDataNameMatrix[i].substr(0,pos);
			unsigned int num = atoi(_trainDataNameMatrix[i].substr(pos+1).c_str());
			//cout<<"num"<<num<<endl;
			for(int j =(_frameRange*(-1));j<=_frameRange;j++){
				int k = j;
				//cout<<"j:"<<j<<endl;
				if(num+j<1||(i+j)>=_numOfTrainData){
					k=0;	
				}
				else {
					unsigned int pos2 = tempTrainDataNameMatrix[i+j].find_last_of("_");
					unsigned int num2 = atoi(tempTrainDataNameMatrix[i+j].substr(pos2+1).c_str());	
				
					if(num2!=(num+j))	k=0;
				}	
				
	
				for(int l=0;l<_featureDimension;l++){
				_trainDataMatrix[i][_featureDimension*(j+_frameRange)+l]=tempTrainDataMatrix[i+k][l];	
				}
			}
			
		}
		
		
		fin.close();

		 dataCount = 0;
		 count = 0;
		 split = 0;
		string* tempTestDataNameMatrix = new string[data.testDataNum];
		float** tempTestDataMatrix = new float*[data.testDataNum];		
		for(int i=0;i<data.testDataNum;i++){
			tempTestDataMatrix[i] = new float [data.inputDim];
		}
		
		ifstream finTest(data.testPath);
		if(!finTest) cout<<"Can't open the test data!\n";
		else cout<<"Inputting test data!\n";
		//string s, tempStr;
		while(getline(finTest,s)&&count<data.testDataNum){
			count++;
			size_t pos = s.find(" ");
			 size_t initialPos=0;
			split=0;
			string tmpName;
			while(split<data.inputDim+1){
				dataCount++;
				split++;
				
				tempStr= s.substr(initialPos, pos-initialPos);
				if (split==1){
					*(tempTestDataNameMatrix+count-1) = tempStr;
				}

				else{
					tempTestDataMatrix[count-1][split-2] = atof(tempStr.c_str());
				}		
				initialPos = pos+1;
				pos=s.find(" ", initialPos);
			}		
			
		}	
		
		
		//fin.close();
	
		_testDataNameMatrix = new string[data.testDataNum];
		_testDataMatrix = new float*[data.testDataNum];
		
		for(int i=0;i<data.testDataNum;i++){
			_testDataMatrix[i]=new float[data.inputDim*(2*_frameRange+1)];
		}
		for(int i=0;i<data.testDataNum;i++){
			unsigned int pos = (*(tempTestDataNameMatrix+i)).find_last_of("_");				
			//cout<<*(tempTrainDataNameMatrix+i)<<endl;
			_testDataNameMatrix[i]=tempTestDataNameMatrix[i];
			string str = _testDataNameMatrix[i].substr(0,pos);
			unsigned int num = atoi(_testDataNameMatrix[i].substr(pos+1).c_str());
			//cout<<"num"<<num<<endl;
			for(int j =(_frameRange*(-1));j<=_frameRange;j++){
				int k = j;
				//cout<<"j:"<<j<<endl;
				if(num+j<1||(i+j)>=_numOfTestData){
					k=0;	
				}
				else {
					unsigned int pos2 = tempTestDataNameMatrix[i+j].find_last_of("_");
					unsigned int num2 = atoi(tempTestDataNameMatrix[i+j].substr(pos2+1).c_str());	
				
					if(num2!=(num+j))	k=0;
				}	
				
	
				for(int l=0;l<_featureDimension;l++){
				_testDataMatrix[i][_featureDimension*(j+_frameRange)+l]=tempTestDataMatrix[i+k][l];	
				}
			}
		}
		finTest.close();
		
		cout << "inputting training label file:\n";
		size_t countLabel  = 0, labelDataCount = 0, numForLabel=0;
		 split = 0;	
		
		_labelMatrix = new int[data.labelDataNum]; 

		ifstream finLabel(data.labelPath);
		if(!finLabel) cout<<"Can't open this file!!!\n";
		string sLabel, tempStrLabel, preLabel= "" ;
		while(getline(finLabel, sLabel)){
			countLabel++;

			unsigned int pos  = sLabel.find(",");
			unsigned int initialPos = 0;
			split=0;
			string tmpName;
			while(split<2){
				labelDataCount++;
				split++;
			
				tempStrLabel = sLabel.substr(initialPos, pos-initialPos);
	//			cout<<"tempStrLabel: "<<tempStrLabel<<endl;
				if (split == 1) tmpName = tempStrLabel;

				if (split==2){
				if(tempStrLabel.compare(preLabel)!=0){
					if(_labelMap.find(tempStrLabel)==_labelMap.end()){
					numForLabel++;
					_labelMap.insert(pair<string, int>(tempStrLabel, numForLabel));	
					}
					preLabel = tempStrLabel;
				}

			
				*(_labelMatrix+countLabel-1)=_labelMap.find(tempStrLabel)->second;
			}
			initialPos = pos+1;
			pos=sLabel.find(",", initialPos);
		}		
	}		
	//cout<<countLabel<<endl;
	//cout<<labelDataCount<<endl;
	
	finLabel.close();	
			
		
		//destructor
		if(_numOfTrainData!=0) delete [] tempTrainDataNameMatrix;
		if(tempTrainDataMatrix!=NULL){
			for(int i =0;i<_numOfTrainData;i++)
				delete tempTrainDataMatrix[i];
		}
		if(_featureDimension!=0) delete []tempTrainDataMatrix;
	
		if(_numOfTestData!=0) delete [] tempTestDataNameMatrix;
		if(tempTestDataMatrix!=NULL){
			for(int i =0;i<_numOfTestData;i++)
				delete tempTestDataMatrix[i];
		}
		if(_featureDimension!=0) delete []tempTestDataMatrix;
		break;
		}
	default:
		cout<<"No match mode!"<<endl;
		break;
	}
};
*/
Dataset::Dataset(const Dataset& d){
	_featureDim = d._featureDim;
	_dataNum = d._dataNum;
	_labelNum = d._labelNum;
	_frameRange = d._frameRange;
	_batchCtr = d._batchCtr;
	_notOrig = d._notOrig;
	_name = new string[_dataNum];
	_data = new float*[_dataNum];
	_label= new int   [_dataNum];
	for (int i = 0; i < _dataNum; i++){
		_name[i] = d._name[i];
		_label[i] = d._label[i];
		_data[i] = new float[_featureDim];
		for (int j = 0; j < _featureDim; j++){
			_data[i][j] = d._data[i][j];
		}
	}
};

Dataset::~Dataset(){
	if (_name != NULL)
		delete[] _name;
	if (_label != NULL)
		delete[] _label;
	if (_notOrig == false){
		for(int i = 0; i < _dataNum; i++){
			delete[] _data[i];
		}
	}
	delete[] _data;
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
	// for debugging, print both matrices
	/*
	cout << "This is the feature matrix\n";
	batch.print();
	cout << "from trainX pointer:\n";
	prtPointer(batchFtre, _numOfLabel, batchSize);
	cout << "This is the label matrix\n";
	batchLabel.print();
	*/
}
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
}

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
			//*(inputReshaped + i*r + j) = *(*(input + i) +j);
			*(inputReshaped + i*r + j) = input[i][j];
		}
	}
	mat outputMat(inputReshaped, r, c);
	delete[] inputReshaped;
	inputReshaped = NULL;
	return outputMat;
}
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

