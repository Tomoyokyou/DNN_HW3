#include <iostream>
#include "dataset.h"
#include "cstring"
#include "rnn.h"
#include "util.h"
#include <vector>
#include <ctime>
#include <cstdlib>
using namespace std;
int main(){
	//string featurePath = "/home/larry/Documents/MLDS/DNN_HW3/model/word_vector.txt";
	//string sntPath = "/home/jason/training_pre4.txt";
	//string classPath = "/home/larry/Documents/MLDS/DNN_HW3/model/classes.txt";
	Dataset d(featurePath.c_str(), classPath.c_str(), sntPath.c_str());
	d.parseTestData(testPath.c_str());
	cout << d.getTestSentNum() << endl;
	vector<size_t>dim;
	dim.push_back(200);
	dim.push_back(50);
	dim.push_back(120);
	RNN rnn(0.02,0.0,0.00001,0.25,NORMAL,dim,ALL, 5,d);
	rnn.train(d,20,0.8,0.99);
	//RNN rnn(0.01,0.4,0.0001,1,NORMAL,dim,ALL, 5,d);
	//rnn.train(d,50,0.8,0.99);
	rnn.save("model/out.mdl");

}
