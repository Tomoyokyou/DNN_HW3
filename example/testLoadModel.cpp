#include <iostream>
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
	Dataset d(featurePath.c_str(), classPath.c_str(), sntPath.c_str(), 55);
	d.parseTestData(testPath.c_str());

	RNN rnn;
	rnn.load(fn);
	cout << "End of load file!\n";
	//rnn.predict(d, "/home/larry/Documents/MLDS/DNN_HW3/model/rnnPredict.csv");
	rnn.predict(d, "./model/best47.csv");
	//rnn.save("./model/debug.mdl");
	cout << "End of save file!\n";
	return 0;
}
