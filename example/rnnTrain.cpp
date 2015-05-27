#include <iostream>
#include "dataset.h"
#include "cstring"
#include "rnn.h"
#include "util.h"
#include <vector>
#include <ctime>
#include <cstdlib>
using namespace std;
int main(int argc,char** argv){
	//string featurePath = "/home/larry/Documents/MLDS/DNN_HW3/model/word_vector.txt";
	//string sntPath = "/home/jason/training_pre4.txt";
	//string classPath = "/home/larry/Documents/MLDS/DNN_HW3/model/classes.txt";
	//string featurePath = "/home/hui/project/rnnFeat/word_vector.txt";
	//string sntPath = "/home/hui/project/rnnFeat/training_oov.txt";
	//string classPath = "/home/hui/project/rnnFeat/classes.sorted.txt";
	//string testPath = "/home/hui/project/rnnFeat/testing_data_parse2.txt";
	if(argc<5){cerr<<"ERROR: Missing path\n";return 1;}
	string featurePath(argv[1]);
	string sntPath(argv[2]);
	string classPath(argv[3]);
	string testPath(argv[4]);
	
	
	Dataset d(featurePath.c_str(), classPath.c_str(), sntPath.c_str());
	d.parseTestData(testPath.c_str());
	cout <<"testset sentences numbers:"<< d.getTestSentNum() << endl;
	vector<size_t>dim;
	dim.push_back(200);
	dim.push_back(50);
	dim.push_back(120);
	RNN rnn(0.01,0.8,0.0001,1,NORMAL,dim,ALL, 5,d);
	rnn.train(d,50,0.8,0.99);
	rnn.save("model/out.mdl");

}
