#include <iostream>
#include <unordered_map>
#include <fstream>
#include <string>
#include <algorithm>
#include <sstream>

using namespace std;

typedef unordered_map<string, float> StringProbMap;

void parseQuestion(string set[][5], size_t opt, const string& sentence);
int main(int argc, char** argv){
	
	StringProbMap spm;
	char buf[500];
	string qSets[5][5];

	ifstream iftest("/home/jason/testing_data.txt");
	ifstream iflm("/home/larry/Documents/MLDS/DNN_HW3/model/training.lm");
	if(!iftest.is_open()){
		cerr << "Fail to open file: /home/jason/testing_data.txt. \n";
		return 1;
	}
	if(!iflm.is_open()){
		cerr << "Fail to open file: /home/larry/Documents/MLDS/DNN_HW3/model/training.lm. \n";
		return 1;
	}

	while(iflm.getline(buf, sizeof(buf), "\n")){
		string temp(buf);
		if(temp.find_first_of("\\") != string:npos && temp.find_first_of(":") != string::npos){
			size_t ngram = stoi(temp.substr(temp.find_first_of("\\")+1, 1));
			while(iflm.getline(buf, sizeof(buf), "\n")){
				stringstream ss(temp);
				string token;
				string words = "";
				ss >> token;
				float prob = stof(token);
				for(size_t i = 0; i < ngram; i++){
					ss >> token;
					words.append(token);
				}
				spm.insert(std::make_pair<std::string, float>(words, prob));
			}
		}
	}

	cout << "Loading language model completed.\n";

	while(iftest.getline(buf, sizeof(buf), "\n")){
		string temp(buf);
		for(size_t i = 0; i < 5; i++){
			for(size_t j = 0; j < 5; j++){
				qSets[i][j] = "";
			}
		}
		parseQuestion(qSets, 0, temp);
		for(size_t i = 1, i < 5, i++){
			if(iftest.getline(buf, sizeof(buf), "\n")){
				temp.assign(buf);
				parseQuestion(qSets, i, temp);
			}
			else{
				cerr << "Error in parsing questions. \n";
				return 2;
			}
		}
	}

	cout << "Loading testing questions completed.\n";

	return 0;
}

void parseQuestion(string set[][5], size_t opt, const string& sentence){
	size_t fbracket = sentence.find_first_of("[");
	size_t bbracket = sentance.find_first_of("]");
	
	set[opt][2] = sentence.substr(fbracket+1, bbracket-1);

	string front = sentence.substr(0, max(0, fbracket-2));
	string back = sentence.substr(min(bbracket+2, string::npos));
	size_t fSpace = front.find_last_of(" ");
	size_t bSpace = back.find_first_of(" ");

	set[opt][1] = front.substr(fSpace+1);
	set[opt][3] = back.substr(0, bSpace-1);

	front = front.substr(0, max(fSpace-2, 0));
	back = back.substr(min(bSpace+2, string::npos));
	
	fSpace = front.find_last_of(" ");
	bSpace = back.find_first_of(" ");

	set[opt][0] = front.substr(fSpace+1);
	set[opt][4] = back.substr(0, bSpace-1);
}
