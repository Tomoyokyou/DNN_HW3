#include <iostream>
#include <utility>
#include <unordered_map>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <exception>
#include <cfloat>

using namespace std;

typedef unordered_map<string, float> StringProbMap;
typedef pair<string, float> StringProbPair;

void parseQuestion(string set[][5], size_t opt, const string& sentence);
int main(int argc, char** argv){
	
	StringProbMap spm;
	StringProbMap sw;
	char buf[500];
	string qSets[5][5];
	size_t gramNum[3]; 

	ifstream iftest("/home/jason/testing_data.txt");
	ifstream iflm("/home/larry/Documents/MLDS/DNN_HW3/model/training.lm");
	ofstream ofs("/home/larry/Documents/MLDS/DNN_HW3/model/viterbiTrigramIp.csv");
	
	if(!iftest.is_open()){
		cerr << "Fail to open file: /home/jason/testing_data.txt. \n";
		return 1;
	}
	if(!iflm.is_open()){
		cerr << "Fail to open file: /home/larry/Documents/MLDS/DNN_HW3/model/training.lm. \n";
		return 1;
	}
	if(!ofs.is_open()){
		cerr << "Fail to open file: /home/larry/Documents/MLDS/DNN_HW3/model/viterbiTrigram. \n";
		return 1;
	}

	
	while(iflm.getline(buf, sizeof(buf), '\n')){
		string temp(buf);
		size_t headerIdx = temp.find_first_of("\\");
		if(headerIdx != string::npos && temp.find("data") != string::npos){
			for(size_t i = 0; i < 3; i++){
				iflm.getline(buf, sizeof(buf), '\n');
				temp.assign(buf);
				try{
					gramNum[i] = stoi(temp.substr(temp.find_first_of("=")+1));
				}
				catch(exception& e){
					cout << temp << endl;
					cout << e.what() << endl;
					return -1;
				}
				cout << i + 1 << "-gram Num: " << gramNum[i] << endl;
			}
		}
		else if(headerIdx != string::npos && temp.find("gram") != string::npos){
			size_t ngram;
			try{
				ngram = stoi(temp.substr(temp.find_first_of("\\")+1, 1));
			}
			catch(exception& e){	
				cout << temp << endl;
				cout << e.what() << endl;
				return -1;
			}
			for(size_t i = 0; i < gramNum[ngram-1]; i++){
				iflm.getline(buf, sizeof(buf), '\n');
				temp.assign(buf);
				stringstream ss(temp);
				string token;
				string words = "";
				float prob;
				ss >> prob;
				for(size_t j = 0; j < ngram; j++){
					ss >> token;
					words.append(token);
				}
				StringProbPair pair(words, prob);
				spm.insert(pair);
				int c = ss.peek();  // peek character
				if ( c != EOF ){
					float weight;
					ss >> weight;
					StringProbPair pairw(words, weight);
					sw.insert(pairw);
				}
			}
		}
	}
	cout << "Loading language model completed.\n";
	iflm.close();
	
	ofs << "Id,Answer\n";

	float viterbiTable[5][3];
	size_t qCount = 0;
	while(iftest.getline(buf, sizeof(buf), '\n')){
		string temp(buf);
		for(size_t i = 0; i < 5; i++){
			for(size_t j = 0; j < 5; j++){
				qSets[i][j] = "";
			}
			for(size_t j = 0; j < 3; j++){
				viterbiTable[i][j] = -100.0;
			}
		}

		parseQuestion(qSets, 0, temp);
		for(size_t i = 1; i < 5; i++){
			if(iftest.getline(buf, sizeof(buf), '\n')){
				temp.assign(buf);
				parseQuestion(qSets, i, temp);
			}
			else{
				cerr << "Error in parsing questions. \n";
				return 2;
			}
		}

		/*
		for(size_t i = 0; i < 5; i++){
			for(size_t j = 0; j < 5; j++){
				cout << qSets[i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
		*/

		size_t maxIdx = 0;
		float tempMax = -FLT_MAX;
		for(size_t i = 0; i < 5; i++){
			for(size_t j = 0; j < 3; j++){
				string temp = qSets[i][j];
				temp = temp.append(qSets[i][j+1]).append(qSets[i][j+2]);
				unordered_map<string, float>::const_iterator got = spm.find (temp);
				if ( got != spm.end() ){
					viterbiTable[i][j] = got->second;
				}
				else if(j < 2){
					temp = qSets[i][j+1];
					got = spm.find (temp.append(qSets[i][j+2]));
					if ( got != spm.end() ){
						viterbiTable[i][j] = got->second;
						got = spm.find (qSets[i][j]);
						if ( got != spm.end() ){
							viterbiTable[i][j] += got->second;
						}
						else{
							viterbiTable[i][j] += -7.681074;
						}
					}
				}
				if(viterbiTable[i][j] > tempMax){
					tempMax = viterbiTable[i][j];
					maxIdx = i;
				}
				cout << viterbiTable[i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;

		cout << "Max: " << (char)('a' + maxIdx) << endl;
		ofs << ++qCount << "," << (char)('a' + maxIdx) << endl;
	}

	cout << "Loading testing questions and viterbi algorithm completed.\n";

	return 0;
}

void parseQuestion(string set[][5], size_t opt, const string& sentence){
	int fbracket = sentence.find_first_of("[");
	int bbracket = sentence.find_first_of("]");
	
	set[opt][2] = sentence.substr(fbracket+1, (bbracket-1)-(fbracket+1) + 1);

	string front = sentence.substr(0, fbracket-1);
	string back = sentence.substr(bbracket+2);


	int fSpace = front.find_last_of(" ");
	int bSpace = back.find_first_of(" ");

	set[opt][1] = front.substr(fSpace+1);
	set[opt][3] = back.substr(0, bSpace);

	front = front.substr(0, fSpace);
	back = back.substr(bSpace+1);
	
	fSpace = front.find_last_of(" ");
	bSpace = back.find_first_of(" ");

	set[opt][0] = front.substr(fSpace+1);
	set[opt][4] = back.substr(0, bSpace);
	
}
