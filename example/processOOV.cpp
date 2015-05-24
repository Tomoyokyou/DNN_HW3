#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <unordered_map>

using namespace std;

typedef unordered_map<string, size_t> StringClassMap;
typedef pair<string, size_t> StringClassPair;

int main(int argc, char** argv){
	if(argc != 2){
		cerr << "Invalid number of arguments!\n";
		exit(1);
	}
	string articlePath(argv[1]);
	string classPath("/home/larry/Documents/MLDS/DNN_HW3/model/classes.sorted.2.txt");
	string outputPath("/home/larry/Documents/MLDS/DNN_HW3/model/training_oov.txt");

	ifstream ifclass(classPath);
	if(!ifclass.is_open()){
		cerr << "Fail to open file: " << classPath << " !\n";
	}
	ofstream ofs(outputPath);
	if(!ofs.is_open()){
		cerr << "Fail to open file: " << classPath << " !\n";
	}

	StringClassMap scm;
	char buf[10000];
	stringstream ss;

	while(ifclass.getline(buf, sizeof(buf))){
		string temp(buf);
		string word;
		size_t id;
		ss.clear();
		ss.str(temp);
		ss >> word >> id;
		StringClassPair scp(word, id);
		scm.insert(scp);
	}
	cout << "Load class of words.\n";
	ifclass.close();

	ifstream ifs(articlePath);
	if(!ifs.is_open()){
		cerr << "Fail to open file: " << articlePath << " !\n";
	}

	
    //cout <<	ifs.getline(buf, sizeof(buf)) << endl;
	//string temp(buf);
	//cout << temp << endl;

	bool isHead = true;

	while( ifs.getline(buf, sizeof(buf))){
		string temp(buf);
		ss.clear();
		ss.str(temp);
		while(!ss.eof()){
			string word;
			ss >> word;
			unordered_map<string, size_t>::iterator it = scm.find(word);
			if(it != scm.end()){
				if(isHead == true){
					ofs << word;
					isHead = false;
				}
				else{
					ofs << " " << word;
				}
			}
			else{
				ofs << " <unk>";
			}
		}
		ofs << endl;
		isHead = true;
	}

	cout << "Finished changing infrequent vocs to <unk>\n";

	ifs.close();
	ofs.close();
}
