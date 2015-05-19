#include <iostream>
#include <string>
#include "rnn.h"

using namespace std;

int main(int argc, char** argv){
	string fn(argv[1]);
	RNN rnn;
	rnn.load(fn);
	cout << "End of load file!\n";
	rnn.save("./model/debug.mdl");
	cout << "End of save file!\n";
	return 0;
}
