#include <iostream>
#include "dataset.h"
#include "cstring"
#include "rnn.h"
#include "util.h"
#include "parser.h"
#include <vector>
#include <ctime>
#include <cstdlib>
using namespace std;
void myUsage(){
	cerr<<"$rnnTrain [feature] [sentence] [class] [test] --[options]"<<endl;
	cerr<<"---------------------------------------------------------"<<endl;
	cerr<<"Options:"<<endl;
	cerr<<"[--rate]:learning rate for RNN training"<<endl;
	cerr<<"[--momentum]:momentum for RNN tranining"<<endl;
	cerr<<"[--var]:variance initializing RNN weights"<<endl;
	cerr<<"[--step]:Backpropagation steps number"<<endl;
	cerr<<"[--epoch]:maximum epoch numbers"<<endl;
	cerr<<"[--outF]:writen RNN model to specific directory"<<endl;
	cerr<<"[--decay]:decay coefficient for learning rate in RNN training"<<endl;
	cerr<<"[--reg]:regularization coefficient in calculation of gradient descent"<<endl;
	cerr<<"[--hidden]:specify dimension for hidden layer"<<endl;
}
int main(int argc,char** argv){
	//string featurePath = "/home/larry/Documents/MLDS/DNN_HW3/model/word_vector.txt";
	//string sntPath = "/home/jason/training_pre4.txt";
	//string classPath = "/home/larry/Documents/MLDS/DNN_HW3/model/classes.txt";
	//string featurePath = "/home/hui/project/rnnFeat/word_vector.txt";
	//string sntPath = "/home/hui/project/rnnFeat/training_oov.txt";
	//string classPath = "/home/hui/project/rnnFeat/classes.sorted.txt";
	//string testPath = "/home/hui/project/rnnFeat/testing_data_parse2.txt";
	srand((unsigned)time(NULL));
	PARSER p;
	p.addMust("FeatureFileName",false);
	p.addMust("SentenceFileName",false);
	p.addMust("ClassFileName",false);
	p.addMust("TestFileName",false);
	p.addOption("--rate",true);
	p.addOption("--momentum",true);
	p.addOption("--epoch",true);
	p.addOption("--decay",true);
	p.addOption("--var",true);
	p.addOption("--step",true);
	p.addOption("--reg",true);
	p.addOption("--hidden",true);
	p.addOption("--outF",false);
	p.addOption("--ans",false);
	string featurePath,sntPath,classPath,testPath,outF,ansF;

	float rate,momentum,decay,var,reg;
	size_t epoch,step,hidden;
	if(!p.read(argc,argv)){
		myUsage();
		return 1;
	}
	p.getString("FeatureFileName",featurePath);
	p.getString("SentenceFileName",sntPath);
	p.getString("ClassFileName",classPath);
	p.getString("TestFileName",testPath);
	if(!p.getNum("--rate",rate))rate=0.01;
	if(!p.getNum("--epoch",epoch))epoch=50;
	if(!p.getNum("--decay",decay))decay=0.98;
	if(!p.getNum("--var",var))var=0.2;
	if(!p.getNum("--step",step))step=3;
	if(!p.getNum("--hidden",hidden))hidden=50;
	if(!p.getNum("--momentum",momentum))momentum=0;
	if(!p.getNum("--reg",reg))reg=0.0001;
	if(!p.getString("--outF",outF))outF="./out.mdl";
	if(!p.getString("--ans",ansF))ansF="";
	p.print();
	
	Dataset d(featurePath.c_str(), classPath.c_str(), sntPath.c_str());
	d.parseTestData(testPath.c_str());
	cout <<"testset sentences numbers:"<< d.getTestSentNum() << endl;
	vector<size_t>dim;
	dim.push_back(d.getFeatureDim());
	dim.push_back(hidden);
	dim.push_back(d.getClassNum());
	RNN rnn(rate,momentum,reg,var,NORMAL,dim,ALL, step,d);
	rnn.train(d,epoch,0.8,decay);
	rnn.save(outF);

}
