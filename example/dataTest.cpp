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
	string featurePath = "/home/larry/Documents/MLDS/DNN_HW3/model/word_vector.txt";
	string sntPath = "/home/jason/training_pre4.txt";
	string classPath = "/home/larry/Documents/MLDS/DNN_HW3/model/classes.txt";
	Dataset d(featurePath.c_str(), classPath.c_str(), sntPath.c_str());
	vector<size_t>dim;
	dim.push_back(200);
	dim.push_back(100);
	dim.push_back(500);
	RNN rnn(0.001,0.1,0,0.01,NORMAL,dim,ALL, 5);
	rnn.train(d,20,0.8,0.99);
	rnn.save("out.mdl");

}
