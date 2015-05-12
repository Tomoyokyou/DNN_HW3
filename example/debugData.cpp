#include<iostream>
#include "dataset.h"

using namespace std;

int main(){
	const char* trainPath = "/home/hui/model/train/train_gender_norm.svm";
	Dataset allData(trainPath);
	Dataset trainData;
	Dataset validData;
	allData.dataSegment(trainData, validData);

	cout << trainData.getDataNum() << endl;
	cout << trainData.getFeatureDim() << endl;
	vector<size_t> label = trainData.getLabel_vec();
	cout << "vector size is: " << label.size();
	mat label_mat = trainData.getLabel_mat();
	//label_mat.print();
	mat batch;
	vector<size_t> bLabel;
	for(int i = 0; i < 230; i ++){
		validData.getRecogData(10, batch, bLabel);
		if (i = 10){
			batch.print();
			for (int j= 0; j < bLabel.size(); j++){
				cout <<bLabel[j] << " ";
			}
			cout <<endl;
		}
	}

	

}
