#include <iostream>
#include "dataset.h"
#include "cstring"
using namespace std;
int main(){
	string featurePath = "/home/larry/Documents/MLDS/DNN_HW3/model/word_vector.txt";
	string sntPath = "/home/jason/training_pre4.txt";
	string classPath = "/home/larry/Documents/MLDS/DNN_HW3/model/classes.txt";
	Dataset d(featurePath.c_str(), classPath.c_str(), sntPath.c_str());
}
