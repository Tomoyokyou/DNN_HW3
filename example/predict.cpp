/*#include <iostream>
#include <string>
#include "rnn.h"
#include "dataset.h"

using namespace std;

int main(int argc, char** argv){
	string fn(argv[1]);
	string featurePath = "/home/ahpan/DNN_HW3/model/preprocess_3/word_vector.txt";
	string sntPath = "/home/ahpan/DNN_HW3/model/preprocess_3/training_oov.txt";
	string classPath = "/home/ahpan/DNN_HW3/model/preprocess_3/classes.sorted.txt";
	string testPath = "/home/ahpan/DNN_HW3/model/preprocess_3/testing_data_parse2.txt";
	Dataset d(featurePath.c_str(), classPath.c_str(), sntPath.c_str());
	d.parseTestData(testPath.c_str());

	RNN rnn;
	rnn.load(fn);
	cout << "End of load file!\n";
	//rnn.predict(d, "/home/larry/Documents/MLDS/DNN_HW3/model/rnnPredict.csv");
		rnn.predict(d, "./model/test.csv");
	//rnn.save("./model/debug.mdl");
	cout << "End of save file!\n";
	return 0;
}*/
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
	cerr<<"$rnnTrain [feature] [sentence] [class] [test] [model] --[options]"<<endl;
	cerr<<"---------------------------------------------------------"<<endl;
	cerr<<"Options:"<<endl;
	/*cerr<<"[--rate]: learning rate for RNN training"<<endl;
	cerr<<"[--momentum]: momentum for RNN tranining"<<endl;
	cerr<<"[--var]: variance initializing RNN weights"<<endl;
	cerr<<"[--step]: Backpropagation steps number"<<endl;
	cerr<<"[--epoch]: maximum epoch numbers"<<endl;
	*/
	cerr<<"[--outF]: writen RNN model to specific directory"<<endl;
	/*cerr<<"[--decay]: decay coefficient for learning rate in RNN training"<<endl;
	cerr<<"[--reg]: regularization coefficient in calculation of gradient descent"<<endl;
	cerr<<"[--hidden]: specify dimension for hidden layer"<<endl;
	*/
	cerr<<"[--cutClass]: specify class number to drop"<<endl;
	//cerr<<"[--hidnum]: numbers of hidden layers"<<endl;
	//cerr<<"[--load]: load rnn model and continuing training"<<endl;
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
	p.addMust("ModelFileName",false);
	//p.addOption("--rate",true);
	//p.addOption("--momentum",true);
	//p.addOption("--epoch",true);
	//p.addOption("--decay",true);
	//p.addOption("--var",true);
	//p.addOption("--step",true);
	//p.addOption("--reg",true);
	//p.addOption("--hidnum",true);
	p.addOption("--cutClass", true);
	//p.addOption("--hidden",true);
	p.addOption("--outF",false);
	//p.addOption("--load",false);
	string featurePath,sntPath,classPath,testPath,outF,loadF;

	//float rate,momentum,decay,var,reg;
	int cutClass;
	//size_t epoch,step,hidden,hidnum;
	if(!p.read(argc,argv)){
		myUsage();
		return 1;
	}
	p.getString("FeatureFileName",featurePath);
	p.getString("SentenceFileName",sntPath);
	p.getString("ClassFileName",classPath);
	p.getString("TestFileName",testPath);
	p.getString("ModelFileName",loadF);
	/*
	if(!p.getNum("--rate",rate))rate=0.01;
	if(!p.getNum("--epoch",epoch))epoch=50;
	if(!p.getNum("--decay",decay))decay=0.98;
	if(!p.getNum("--var",var))var=0.2;
	if(!p.getNum("--step",step))step=3;
	if(!p.getNum("--hidden",hidden))hidden=50;
	if(!p.getNum("--momentum",momentum))momentum=0;
	if(!p.getNum("--reg",reg))reg=0.0001;
	*/
	if(!p.getNum("--cutClass",cutClass))cutClass=50;
	//if(!p.getNum("--hidnum",hidnum))hidnum=1;
	if(!p.getString("--outF",outF))outF="./out.mdl";
	p.print();
	
	Dataset d(featurePath.c_str(), classPath.c_str(), sntPath.c_str(), cutClass);
	d.parseTestData(testPath.c_str());
	//cout << "cutClass is " << cutClass << endl;
	//cout <<"testset sentences numbers:"<< d.getTestSentNum() << endl;
		cout<<"Loading RNN model..."<<endl;
		RNN loadM;
		loadM.load(loadF);
		cout<<"Predicting..."<<endl;
		loadM.predict(d,outF);
		cout<<"result witen in:[ "<<outF<<"]"<<endl;
	return 0;
}
